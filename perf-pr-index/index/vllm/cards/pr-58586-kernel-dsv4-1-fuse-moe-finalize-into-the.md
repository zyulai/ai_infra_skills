# PR #58586: [Kernel][DSV4.1] Fuse MoE finalize into the TP all-reduce + mHC boundary

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-09-26 |
| 分类 | MoE \| tier: major |
| Label | ready, ci/build, deepseek, nvidia, quantization, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注「5 file(s) dropped from diff for size (see PR for full diff)」 |
| 链接 | https://github.com/vllm-project/vllm/pull/58586 |

## 一句话结论

把 MoE finalize 与 shared add 折进 all-reduce+mHC 边界，DSV4.1-Flash TP4 吞吐 +1.8~1.9%（c≤8）。

## 动机

DSV4.1 decode（TP4、0 < M <= 16、FULL CUDA graphs）每个子层以 fused TP all-reduce + mHC kernel 收尾；MoE 之后该边界前还要执行 FlashInfer 的 MoE finalize（top-k 加权和）与一次独立的 shared-expert add。本 PR 把两者折进边界，省去多余 kernel 与访存。

## 优化点

Vendored FlashInfer 的 CuTe DSL low-latency MNNVL all-reduce（H5120 presets）：`_SharedOnlyPublishDeviceKernel`、`_group_leader_block_sum` 与上游逐字节一致；`_QuadFinalizePublishDeviceKernel` 增加 `fp32_weights` 选项以读取 DSV4.1 router 的 FP32 top-k 权重（上游为 BF16）；`_LamportMHCDeviceKernel` 把 residual add/copy 换成 hc streams 的 mHC post-mix 与 pre-mix collapse，再由未改动的 RMSNorm 归一化，替换 overlap 路径上的 `torch.ops._C_custom_ar.all_reduce_mhc`（#57643）并删除其 CUDA 文件。MoE 侧经 #58635 的 deferral 保持 unfinalized（`defer_moe_finalize`、`should_defer_moe_finalize` 返回 `UnfinalizedMoEOutput`）。DSV4.1-Flash 40 层中 37 层 defer，`mhc_shifted_post_pre` 调用 `AllReduceMHC.finalize`，使 top-k reduce、shared add、all-reduce、mHC post/pre 成为两个 PDL-chained kernels；FULL graphs 之外改为 inline。

## 关键改动

- `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py`（新增）：vendored + 改造的 CuTe DSL LL kernels，含 fp32_weights finalize 与 mHC post-mix/collapse/RMSNorm 融合。
- `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py`：导出上述内核。
- `vllm/models/deepseek_v41/nvidia/model.py`：DSV4.1 wiring，37/40 层 defer、`forward_unfinalized`、`mhc_shifted_post_pre` 调用 finalize。
- `csrc/libtorch_stable/all_reduce_mhc.cu`（删除）：旧 #57643 CUDA kernel，无其他使用者。

```text
"""Lamport all-reduce fused with DSV4.1's mHC post, collapse and RMSNorm.

FlashInfer's LL kernels, copied: a publish kernel (or its MoE finalize
variant) multicasts each rank's contribution into a three-generation Lamport
mailbox, and a PDL-chained consumer spins until every rank's fragment lands,
reduces it and normalizes. The consumer's residual add and copy are replaced by
the mHC post-mix of the hc residual streams and their pre-mix collapse, which
the RMSNorm then normalizes. The finalize variant also reads FP32 routing
weights.
"""
```

## 性能声明

条件：DSV4.1-Flash、TP4 GB300、FULL CUDA graphs，flashinfer 0.7.0 / nvidia-cutlass-dsl 4.7.1 / torch 2.13.0+cu130；`vllm bench serve` random 1024 in / 256 out、`--ignore-eos`、并发 1/4/8/16；baseline 为同树关闭 deferral。E2E output tok/s：并发 1 为 174.7 → 177.8（+1.8%）、4 为 581.7 → 591.7（+1.7%）、8 为 1069.7 → 1089.8（+1.9%）、16 为 1818.6 → 1817.6（flat）；median ITL 5.14 → 5.03 / 5.71 → 5.56 / 6.30 → 6.17 / 7.19 → 7.13 ms。正文另称 pre-rebase sweep 在 c=16 测得 +3.0% throughput。Kernel level（µs，单层两边界，graph 8→40 次调用斜率、ranks 取 max、50 次 replay 中位数）：M=1 时 48.59 → 45.28（Δ −3.31）、M=4 82.50 → 79.22（−3.28）、M=8 117.61 → 111.61（−6.00）、M=16 186.22 → 180.26（−5.96）。Accuracy：GSM8K（5-shot greedy、n=1319、并发 16、各 3 次）均值 0.9285 → 0.9267；GPQA-Diamond exact match / pass@4 0.9040 / 0.9545 → 0.9066 / 0.9444。

## 验证与回归风险

验证：`tests/distributed/test_custom_all_reduce.py::test_all_reduce_mhc_matches_unfused_path`（替代 #57643 的 CUDA-kernel 测试；M∈{1,2,3,4,5,6,8,12,16}、shrinking/growing 顺序、±0 sentinel、CUDA graph replay、padded/permuted rows 与 −1 route；4× GB300 该文件 51 个测试全通过）、`pre-commit run`、FULL graphs 下 in-engine 数值检查（GSM8K 每 rank 51,393 次 fused boundary，0/1.6e10 元素不一致）、E2E perf 与精度。可能受影响路径：37 个 defer 层、M > 16 路径（正文称不变）、eager/PIECEWISE、mHC 与 RMSNorm 数值（FlashInfer RMSNorm 求和顺序与近似 rsqrt，1 BF16 ULP 内差异）；正文称 GSM8K 0.18 pt 差距"unexplained"，疑与 timing-dependent batching 有关。

## 关联

正文明确关联：#58635（merged，提供 modular-path deferral）、#57643（提供 boundary wiring，其 CUDA kernel 在本 PR 被删）、#52601（MiniMax-M3 的 MoE tail all-reduce fusion，目标模型与 kernel 不同，仅作 duplicate check）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
