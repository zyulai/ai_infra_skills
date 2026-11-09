# PR #58584: [ROCm][Perf][GLM-5.3-Flash] BF16 splitk kernel for sparse MLA decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simondanielsson |
| 合并 | 2026-10-08 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, rocm, ready, glm |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/58584 |

## 一句话结论

bf16 sparse MLA decode 改用 splitK+reduce，MI350 上 ITL 降低 5-15%、topk=2048 微基准 6x 加速。

## 动机

PR 正文 Purpose：bf16 sparse MLA 路径的 decode rows 走的是 ragged **prefill** kernel，其 grid 为 `(q_len, heads_blocks)`。GLM-5.3-Flash 在 TP8 下只有一个 head block，decode 并发 4 时仅在 256 个 CU 上启动 4 个 workgroup，明显欠利用。新 kernel 改为在 context 维做 splitK+reduce，grid 变为 `(q_len, num_kv_splits, heads_blocks)`（正文称 DSv4 有类似实现）。当 `topk>=512` 时启用（GLM 使用 topk=2048），因微基准显示更小 topk 下 splitK 反而更慢。

## 优化点

新增 BF16 splitK decode kernel 及配套调度：把原本按 `(q_len, heads_blocks)` 的 ragged prefill 网格扩展为按 KV/context 维拆分的 `(q_len, num_kv_splits, heads_blocks)`，再做 reduce 合并，从而在多 CU 上并行 decode；以 `topk>=512` 作为启用门槛，避开小 topk 下 splitK 的劣势。改动落在 `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`（backend 选择/启用条件）与 `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`（kernel 实现），并新增 `benchmarks/kernels/benchmark_sparse_mla_decode_bf16.py` 微基准；相关测试见 `tests/kernels/attention/test_rocm_aiter_mla_sink.py`、`test_rocm_triton_attn_dsv4.py`。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: 新增 BF16 splitK+reduce sparse MLA decode kernel
vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py: 按 topk>=512 选择 splitK 路径
benchmarks/kernels/benchmark_sparse_mla_decode_bf16.py: 新增 splitK vs ragged 微基准脚本
tests/kernels/attention/test_rocm_aiter_mla_sink.py、test_rocm_triton_attn_dsv4.py: 相关 kernel 测试
```

## 性能声明

正文标题式声明："Perf: e2e -5-15% ITL. Microbenchmarks: 6x speedup on topk=2048"。e2e（MI350 除非另注）：serve `zai-org/GLM-5.3-Flash --tensor-parallel-size 8 --max-num-seqs 512 --gpu-memory-utilization 0.8 --attention-backend ROCM_AITER_MLA_SPARSE --max-num-batched-tokens 1638`，`VLLM_ROCM_USE_AITER=1`。MI350 1K/1K conc32：TPOT 14.35 → 12.88 ms（-10.24%）、ITL 14.06 → 12.66 ms（-9.96%）、Output tput 2083.73 → 2328.81 tok/s（+11.76%）、TTFT 561 → 560 ms；64K/1K conc4：TPOT 15.57 → 13.32（-14.45%）、ITL 11.89 → 9.92（-16.57%）、Output tput 223.33 → 249.90（+11.90%）；256K/1K conc4 ITL 12.38 → 10.36（-16.32%）；512K/1K conc4 ITL 12.78 → 10.85（-15.10%）。MI300 8k/1k conc32：Output tput 1123.95 → 1227.47 tok/s、Mean TPOT 26.11 → 23.71 ms。Microbenchmark（MI350，`benchmark_sparse_mla_decode_bf16.py --rows 1 2 4 8 16 64 128 256 --lens 128 512 2048 --splits auto`）：topk=128 时 splitK 反而更慢（rows=1：ragged 23.96 us vs split-K 32.78 us，+8.82 us），故该域不启用。精度：gsm8k MI350 flexible 0.9219(main) → 0.9196(this branch)、strict 0.9196 → 0.9174；MI300 flexible 0.9136 → 0.9181；gpqa_diamond_zeroshot acc 0.4747 → 0.4798。

## 验证与回归风险

Test Result 给出 gsm8k（MI350/MI300）、gpqa（未开 thinking mode）精度对比及上述 e2e/microbenchmark。正文注明 main 在 gpqa 的低分可能与 issue #59413 有关（非本 PR 引入）。回归风险：topk=128 等小 topk 域 splitK 变慢，故以 `topk>=512` 门槛规避；e2e 表中 1K/1K conc128 的 TTFT 842 → 843 ms（+0.12%）、512K/1K TTFT 22477 → 22836 ms（+1.60%）等项基本持平或略升。正文未说明对其他模型或非 ROCm 硬件的影响。正文还说明 gpqa 评测未启用 thinking mode，并给出 MI300 上 8k/1k、并发 32 的完整 Serving Benchmark Result 文本。

## 关联

无（正文提及 DSv4 有类似 splitK kernel、gpqa 低分关联 issue #59413，但未提供可对应的 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
