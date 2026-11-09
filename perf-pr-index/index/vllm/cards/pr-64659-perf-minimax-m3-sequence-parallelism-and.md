# PR #64659: [Perf][MiniMax-M3] Sequence parallelism and GEMM-RS for TP + MegaMoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-11-02 |
| 分类 | Speculative Decoding（tier: major） |
| Label | performance, minimax |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 4 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/64659 |

## 一句话结论

MiniMax-M3 在 TP+EP+MegaMoE 下引入 SP 与 GEMM-RS，4×GB300 / 8k in 1k out 下 P50 TTFT 降 14–20%、c=64 输出吞吐 +13.6%。

## 动机

正文 Overview：本 PR 为 NVIDIA 上 TP + EP + MegaMoE 的 MiniMax-M3 加入 sequence parallelism (SP)，并为 `o_proj` 与 dense MLP `down_proj` 加入 GEMM-RS kernel，设计沿用既有 Kimi-K3 与 DeepSeek-V4.1 的 SP。没有 SP 时，MegaMoE 虽已在 TP ranks 间按 token 分片，却把分片写进一个零填充的全尺寸 buffer，随后 RMSNorm 再 all-reduce，因此每层要付 2 次 full all-reduce（attention 后一次、几乎全零的 MoE 输出一次）。

## 优化点

有 SP 时 residual stream 保持按 token 分片：每个 MoE 层在 attention 前做 1 次 all-gather、`o_proj` 后做 1 次 reduce-scatter，MegaMoE 直接消费分片（不再 shard/gather/zero-fill）；3 个 dense 层在 MLP 前后各多一对 all-gather/reduce-scatter。`vllm/models/minimax_m3/nvidia/model.py`：embedding 后分片 hidden states 与 MegaMoE padding mask、末端一次 collective 同时 all-gather 输出与 EAGLE3/DSpark aux hidden states；Attention 做 all-gather 输入 / reduce-scatter 输出；`MiniMaxM3MoE` 的 `use_sequence_parallel` 重命名为 `shard_replicated_input`；AR + Gemma RMSNorm 融合在 SP 下跳过。`GemmRsAr.apply()` 现在接受 fused dense GEMM1 输出的 MXFP8 `QuantizedActivation` 并跳过自身量化。SP 在 TP>1 + EP + MegaMoE + PP1 时自动启用，GEMM-RS 通过 `VLLM_ENABLE_GEMM_RS=1` 可选开启，且仅在 M≥1024 使用。

## 关键改动

```text
vllm/models/minimax_m3/nvidia/model.py: SP 主逻辑（embedding 分片、attention/MoE/dense MLP 的 all-gather / reduce-scatter、跳过 AR+RMSNorm 融合）
vllm/model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py: GEMM-RS 支持预量化 MXFP8 输入
benchmarks/kernels/benchmark_gemm_rs_ar.py: 增加 minimax_m3 的 o_proj / down_proj 投影配置
tests/kernels/test_gemm_rs_ar.py: 扩展预量化 MXFP8 输入用例（两种 MXFP8 backend）
```

```diff
+# Smallest batch whose o_proj/down_proj runs the fused GEMM-RS kernel. Below
+# it, the GEMM followed by sp_reduce_scatter is as fast (TP4 GB300, see
+# benchmarks/kernels/benchmark_gemm_rs_ar.py --model minimax_m3).
+_GEMM_RS_MIN_TOKENS = 1024
```

## 性能声明

测试条件：单节点 4×GB300，base `f81a414fae`，`MiniMaxAI/MiniMax-M3.1-Flash`（`9e7ebac`）；serve 命令为 vLLM recipe 的 "Single node (TP4, NVIDIA GB200)" 命令加 `--no-enable-prefix-caching`；对比 TP4、TP4 + EP、TP4 + EP + SP（`VLLM_ENABLE_GEMM_RS=1`）三种 setup；E2E `vllm bench serve --dataset-name speed-bench --speed-bench-config throughput_8k`（8k in / 1k out、`--sweep-max-concurrency 1,4,16,64`），全 sweep 每 setup 跑两次取范围；DSpark spec decode。

| Concurrency | Output throughput TP4 / TP4+EP / TP4+EP+SP (tok/s) | P50 TTFT TP4 / TP4+EP / SP (ms) |
|---|---|---|
| 1 | 410–411 / 370 / 355 | 164–166 / 140 / 111–114 |
| 4 | 1,022–1,058 / 928–950 / 945–972 | 190–193 / 163–167 / 134–137 |
| 16 | 2,212–2,245 / 2,254–2,260 / 2,397–2,410 | 200–202 / 179–184 / 151–152 |
| 64 | 3,937–3,938 / 4,299–4,338 / 4,896–4,914 | 289–329 / 266–280 / 233–236 |

正文 Claims："P50 TTFT is 14-20% lower at all concurrencies"；"Output throughput is +6.5% at concurrency 16 and +13.6% at concurrency 64, a tie at concurrency 4, and -4% at concurrency 1"；"gsm8k is unchanged"。并说明 c=1 时 TPOT 高 3% 源于 spec decode acceptance（3.10 vs 3.20）而非更慢的 step（TPOT × acceptance length 为 8.56 ms vs 8.58 ms）。GEMM-RS microbenchmark（TP4, N=6144, `benchmark_gemm_rs_ar.py`，baseline 为 MXFP8 GEMM + `sp_reduce_scatter`）：M=32768 时 o_proj (K=2048) 952.6→680.1 us (1.40x)、down_proj (K=3072) 1146.9→704.8 us (1.63x)；以预量化输入重测 down_proj M=32768 为 1083.8–1087.4→650.4–652.4 us (1.66–1.67x)。perf claims 段标注 "3% higher"（对应 c=1 TPOT 说明）。

## 验证与回归风险

验证：E2E SpeedBench sweep（见性能声明，每 setup 两次、重启 server）；gsm8k（5-shot、chat completions、temperature 0，每次 server start 一 eval）：flexible-extract TP4 95.53/95.75、TP4+EP 95.45/95.98、SP 96.13/95.45，strict-match 95.00/95.45、95.22/95.68、95.91/95.22；GEMM-RS microbenchmark（含预量化输入重测，M=512–1536 额外重复两次，结论 GEMM-RS 仅用于 M≥1024）；`tests/kernels/test_gemm_rs_ar.py` 扩展预量化 MXFP8 输入，在 2×GB300 通过、CI 在 2×B200 distributed step 运行。正文明确 E2E SP 需要单 NVLink 域内 4 张 SM100 GPU 加 M3.1 checkpoint，CI 不具备，靠上述 benchmark 与 gsm8k 覆盖。限制与后续（Limitations and follow-ups）：MTP blocks 与 PP>1 不用 SP；dense MLP 仍 TP-sharded（仅 3 层）；all-gather 仍搬 BF16，改为 gather MXFP8 `qkv_proj` 输入可减半字节，留作后续。PR 未注明更多回归风险。

## 关联

无（正文提到设计沿用 Kimi-K3、DeepSeek-V4.1 的 SP，但未给出对应 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
