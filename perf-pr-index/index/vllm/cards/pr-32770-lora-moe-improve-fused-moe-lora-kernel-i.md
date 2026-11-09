# PR #32770: [lora/moe] Improve fused MoE‑LoRA kernel indexing and memory access

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cwazai |
| 合并 | 2026-01-26 |
| 分类 | MoE \| tier: major |
| Label | documentation, performance, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/32770 |

## 一句话结论

优化 fused MoE-LoRA kernel 索引访存，Qwen3-30B-A3B 双 LoRA 从 1331.02 升到 1366.80 tok/s。

## 动机

PR 正文说明本 PR 优化 Triton kernel 的 inner-loop indexing 与 memory-access patterns，并列出 Performance impact：hot path 中更快的索引算术、更少的浪费 CUDA 线程（axis-2 early-exit）、更高的 B（expert weights）L2 命中率。此前 kernel 用 int64 索引、列偏移带 `% N` 取模、axis-2 grid 按 `lora_a_stacked[0].shape[0]` 启动（包含未使用的 LoRA adapter），B 矩阵加载也未启用缓存修饰。

## 优化点

正文列出 7 项改动：用 int32/uint32 做索引（PTX 中比 int64 便宜）；移除列偏移的 `% N` 取模，改为直接 `pid_n * BLOCK_SIZE_N` 并用 `cn_mask` / `offs_bn < N` 掩码；`MAX_LORAS_TOTAL` 作为 Python 标量传入（不再用 `tl.num_programs`），kernel 用无符号比较对非法 `lora_id` 早退（自动过滤 -1）；axis-2 program 数由 `lora_a_stacked[0].shape[0]` 降为 `lora_ids.numel()`，只启动实际用到的 LoRA adapter；新增 `USE_B_L2_CACHE=True` 对 B 矩阵加载使用 `.ca` cache modifier 提升重复 expert 权重读的局部性；`expert_id` / `token_id` load 使用 `other=-1` 简化 mask 逻辑；`_fused_moe_lora_shrink` / `_fused_moe_lora_expand` 传入 `lora_a_stacked[0].shape[0]` 并启用 `USE_B_L2_CACHE=True`。

## 关键改动

```text
vllm/lora/ops/triton_ops/fused_moe_lora_op.py: 唯一改动文件，含索引 int32 化、去取模、非法 lora_id 早退、.ca 缓存修饰与 grid 收缩
```

```diff
-    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
+    # remove modulo wrap-around
+    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
     offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
-    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
+    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int32)

-        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
+        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
+        if USE_B_L2_CACHE:
+            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
```

## 性能声明

条件：Hardware Nvidia H800，Software CUDA/driver 13.0，Workload model Qwen3-30B-A3B、rank 64、Two Lora；Serving Benchmark 为 Successful requests 800、Failed requests 0、Maximum request concurrency 80、Total input tokens 819200、Total generated tokens 80000。原文 [before] / [after]：Benchmark duration (s) 60.10 / 58.53；Request throughput (req/s) 13.31 / 13.67；Output token throughput (tok/s) 1331.02 / 1366.80；Peak output token throughput (tok/s) 3125.00 / 3280.00；Peak concurrent requests 110.00 / 113.00；Total Token throughput (tok/s) 14960.72 / 15362.88；Mean TTFT (ms) 642.25 / 616.34；Median TTFT 518.54 / 480.47；P99 TTFT 2830.93 / 3497.42；Mean TPOT (ms) 54.02 / 52.60；Median TPOT 54.39 / 52.30；P99 TPOT 66.80 / 62.77；Mean ITL (ms) 53.48 / 52.07；Median ITL 26.41 / 25.21；P99 ITL 457.24 / 397.00。精度数据 PR 正文未给出。

## 验证与回归风险

Testing：`tests/lora/test_fused_moe_lora_kernel.py` 在改动前后均给出 `48 passed, 14 warnings`，[before] 耗时 `279.58s (0:04:39)`、[after] 耗时 `397.86s (0:06:37)`，正文标注 All pass；[after] `tests/lora/test_olmoe_tp.py` 同样 All pass，`6 passed, 8 warnings in 406.74s (0:06:46)`。diff 中源码侧仅 `vllm/lora/ops/triton_ops/fused_moe_lora_op.py` 一个文件被改动，其中 `_fused_moe_lora_kernel`、`_fused_moe_lora_shrink` 与 `_fused_moe_lora_expand` 三处入口都有改动。回归风险：改动集中在 fused MoE-LoRA Triton kernel 的索引与访存，影响所有走该 kernel 的 MoE + LoRA 路径（正文提及 OLMoE TP 测试）；P99 TTFT 由 2830.93 升到 3497.42 ms，正文未讨论；其它模型、硬件、精度路径 PR 未注明。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
