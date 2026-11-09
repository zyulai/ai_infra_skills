# PR #56638: [ROCm][DSv4.1][Perf] Replace the Torch prefill combine with the shared combine_topk_swa_indices kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JohnQin-llm |
| 合并 | 2026-10-08 |
| 分类 | Attention / Kernel \| tier: major |
| Label | rocm, ready, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/56638 |

## 一句话结论

ROCm DSv4.1 prefill 改用共享 combine kernel，MI355X TTFT 中位 120.7→59.7 ms（-50.5%）。

## 动机

ROCm 的 V4.1 sparse prefill 用普通 Torch indexing 合并 top-k 与 sliding-window 索引，每次约九十个小算子。PR 改为复用 CUDA 路径已在用的共享 `combine_topk_swa_indices` kernel——单次 launch，并顺带带来 SWA bounded-replay bound、SWA-only guard 与 JIT warmup；ROCm 专用 Triton kernel 与 Torch 路径一并删除。

## 优化点

`amd/rocm.py` 的 prefill 改为调用 `common/ops/cache_utils.py` 的 `combine_topk_swa_indices`。共享 kernel 增加两处不改变良构输入的保护：两个长度在 0 处 clamp（warmup batch 可能带负 position，原 Torch 路径有 `clamp_min(0)`）；负的 top-k 索引保持 -1 而非变成 `M*b-1`（会落到前一 request 的行）。删除 ROCm-only Triton kernel 与 Torch 实现。正文强调 CUDA 路径此前已在用该共享 kernel，改动同时把 SWA bounded-replay bound、SWA-only guard 与 JIT warmup 一并带到 ROCm；两处 guard 保证良构输入不受影响。

## 关键改动

```text
vllm/models/deepseek_v41/amd/rocm.py: prefill 改调共享 combine_topk_swa_indices
vllm/models/deepseek_v41/common/ops/cache_utils.py: 共享 kernel 增加长度 clamp(0) 与负索引保留 -1 两处 guard
tests/kernels/attention/test_rocm_triton_attn_dsv4.py: 对照被替换的 Torch 路径、覆盖多种边界并确认 ROCm 派发到共享 kernel
tests/v1/attention/test_deepseek_v4_swa_visible.py: 覆盖 replay bound（含 #59136 的短 replay chunk）
```

```python
+            valid = (topk_indices >= 0) & (topk_indices < N)
             tl.store(
                 combined_indices_ptr + token_idx * combined_indices_stride + offset,
-                topk_indices + M * batch_idx,
+                tl.where(valid, topk_indices + M * batch_idx, -1),
                 mask=mask,
             )
```

## 性能声明

MI355X、TP4、DeepSeek-V4.1-Flash。两臂均跑 `vllm/vllm-openai-rocm:nightly-0cbac6cd1`，B 臂仅挂载本分支两个源文件；每个 slot 为全新 server、2 轮 warmup + 3 轮记录、A B B A 顺序。workload：`vllm serve` TP4/language-model-only/moe-backend aiter/gpu-memory-utilization 0.9/max-model-len 33792/max-num-seqs 128/max-cudagraph-capture-size 1024/max-num-batched-tokens 16384；`vllm bench serve` random、input 8192、output 256、ignore-eos、12 prompts、max-concurrency 2、seed 42。n=6/arm：TTFT median 120.7±1.7→59.7±1.9 ms（-50.5%）、TTFT p99 166.3±2.6→81.7±1.3 ms（-50.9%）、output throughput 225.4±0.2→234.0±0.2 tok/s（+3.8%）、TPOT mean 8.42→8.33 ms（-1.1%）。GSM8K（1319 题、5-shot）：本分支 0.9045，同 session 的 main 为 0.8961 与 0.9067。

## 验证与回归风险

测试（gfx950/MI355X）：`tests/kernels/attention/test_rocm_triton_attn_dsv4.py` 124 passed, 0 failed；`test_v41_combine_topk_swa_stops_at_replay_start` 4 passed, 0 failed。测试把被替换的 Torch 路径原样复制作对照以防漂移，覆盖 seq_len<query_len、seq_len==0、compress_ratio==0、非零 query_start_loc[0]、-1 候选、窗口 64/128/256；两处 guard 分别对应 warmup 批次出现负 position 与失败或越界候选保持 -1 的情形。回归风险：ROCm/DSv4.1 prefill 路径改为共享 kernel。

## 关联

正文提到 #59136（其短 replay chunk 已并入本 PR，故 #59136 不再需要，并注明由 @yuzhouo7 co-authored）；无其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
