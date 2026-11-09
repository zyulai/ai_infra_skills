# PR #48519: [ROCm][Perf] Optimize sparse attention prefill kernel for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kliuae |
| 合并 | 2026-07-16 |
| 分类 | Attention / Kernel(tier: major) |
| Label | rocm, ready, v1, deepseek |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/48519 |

## 一句话结论

ROCm 上 DeepSeek-V4 稀疏 prefill kernel 提速约 2x，E2E TTFT 降约 10%。

## 动机

PR 正文指出，当前 upstream ROCm 路径里 DeepSeek-V4 专用的 sparse attention ragged prefill kernel 存在一些可优化的低效：`num_warps` 设定偏高、加载 `kv` 后又用同一谓词重复 mask、以及没有预取下一 tile。正文说明这些优化针对 DeepSeek-V4-Pro 的运行 shape，合并后 kernel 相比 upstream baseline 约 2x，E2E 下 TTFT 约降 10%。

## 优化点

三处内核级改动。(1) `num_warps` 由 8 改为 4：DSv4-Pro 在 TP8 下 per-workgroup tile 为 `[BLOCK_H=16, 512]` 并归约到 `[16, 16]`，`num_warps` 切分 K 维，切到 8 会带来更大的 cross-wave 归约与更多 LDS 读写 barrier，改为 4 可降低该开销。(2) 去掉冗余 mask：`kv` 加载时已按有效区域 mask，mask 为 False 处元素本就是 `0.0`，紧随其后的 `tl.where` 二次 mask 冗余，且在 load-MFMA 之间插入 mask op 会阻碍 software pipelining。(3) 预取下一 tile：把下一 tile 的 `slot = tl.load(...)` 提前到 `tl.dot` 之前，使下一 tile 的 kv 加载与 MFMA 重叠。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: _sparse_attn_prefill_ragged_kernel 上提 slot 加载、删冗余 mask、预取下一 tile；_rocm_sparse_attn_prefill_ragged_triton 把 num_warps 由 8 改为 4
```

关键片段（删冗余 mask 并预取下一 tile）：

```diff
             mask=valid[:, None] & dim_mask[None, :],
             other=0.0,
         )
-        kv = tl.where(valid[:, None] & dim_mask[None, :], kv, 0.0)
+
+        next_k_pos = k_start + BLOCK_K + k_offsets
+        slot = tl.load(
+            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
+        )
         scores = tl.dot(q, tl.trans(kv)) * scale
```

## 性能声明

Kernel latency（`num_heads=16`、`head_dim=512`，DSv4-Pro TP8）：prefill S=2048 由 0.348 ms 降至 0.190 ms（1.83×）、S=4096 由 1.065 ms 降至 0.526 ms（2.02×）、S=8192 由 2.782 ms 降至 1.306 ms（2.13×），正文称相比 upstream baseline 约 2x speed up。E2E：模型 `deepseek-ai/DeepSeek-V4-Pro`，硬件 MI355、TP8、`--kv-cache-dtype fp8`、`--block-size 256`，`vllm bench serve` random 数据集、ISL/OSL=8k/1k，conc 8~128 下 Total tokens/s +1.6%~+7.3%、mean TTFT -6.7%~-10.5%（conc 16：3202.50→2875.94 ms，-10.2%；conc 128：17251.59→15447.22 ms，-10.5%），正文概括为 TTFT 约降 10%。

## 验证与回归风险

Test Plan 为 `lm_eval` gsm8k 加 MI355 性能 benchmark；gsm8k flexible-extract / strict-match 的 exact_match 分别为 0.9515±0.0059、0.9522±0.0059（无精度回退）。E2E 服务命令为 `vllm serve deepseek-ai/DeepSeek-V4-Pro`、TP8、`--max-num-seqs 512`、`--kv-cache-dtype fp8`、`--block-size 256`、`--no-enable-prefix-caching`、`--compilation-config '{"mode":3,"cudagraph_mode": "FULL_AND_PIECEWISE"}'`，`lm_eval` 用 30-shot、`num_concurrent=128`、`max_gen_toks=2048`。正文的 Test Result 还给出并发 8/16/32/64/128 下 Total tokens/s、Total reqs/s、mean TPOT 与 mean ITL 的逐行前后对比，并附 gsm8k 30-shot 结果用于确认精度未退化。风险：`num_warps=4` 是针对 DSv4-Pro TP8 `[16,16]` tile 的调优，正文未给其他 head_dim、TP 配置或 shape 的数据；改动集中在 ROCm `rocm_aiter_mla_sparse.py` 的稀疏 prefill kernel 与 `VLLM_ROCM_USE_AITER=1` 路径。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
