# PR #43273: [GDN] GDN Prefill kernel for SM100

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-05-26 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, v1, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾标注 7 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/43273 |

## 一句话结论

SM100 新增 CuteDSL GDN prefill kernel，低 V-head/高 TP 下 microbench 对 FLA 最高 5.55×、E2E +22.2%。

## 动机

新增 in-tree 的 SM100 GDN prefill kernel，源自 team CUDA_ERROR_UNKNOWN 为 MLSys 2026 FlashInfer Kernel competition 开发的 CUDA C++ kernel，作者把它移植到 CuteDSL 以获得 JIT 带来的适配多种 head 配置与易开发的好处。该 kernel 需显式 opt-in，默认 backend 仍是 FlashInfer。正文指出 FlashInfer 是单一全融合 kernel、只能跨 request/head 并行，而本 kernel 的 `kkt_inv_uw_kernel` 与 `o_kernel` 可跨所有 chunks 并行，因此在低 head 数（TP）与低 request 数（长 prefill with chunking）下可能更快。

## 优化点

按 FLA 设计拆成 3 个 kernel：`kkt_inv_uw_kernel`（融合 FLA 的 `chunk_scaled_dot_kkt_fwd()`、`solve_tril()`、`recompute_w_u_fwd()`）、`h_kernel`（`chunk_gated_delta_rule_fwd_h()`）、`o_kernel`（`chunk_fwd_o()`）；原生支持 BF16 state，无需额外 BF16→FP32 转换。用 Newton-Schulz 迭代、仅借 tensor cores（`mma.sync`）求 `inv(I + strictlower(A))`，替代 CUDA cores 的前向替换，缓解 GDN 最大瓶颈之一。NCU 显示 kkt 与 H kernel 因无 ping-pong/prefetch 存在 Tensor Core pipeline bubble（矩阵求逆阻塞后续 U/W MMA、需打包 V/H 造成 stall），O kernel 利用率明显更好。

## 关键改动

```text
vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/__init__.py: CuteDSL GDN prefill 的三个 kernel 实现与元数据准备入口
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: 接入 GDN prefill backend 选择
vllm/engine/arg_utils.py: 增加 --gdn-prefill-backend 等启动参数
vllm/cute_utils/（_tcgen05.py、cvt.py 等）: CuteDSL kernel 所需的工具支持
tests/kernels/mamba/test_gdn_prefill_cutedsl.py: 正确性（对 FLA reference 的 MAE）与 microbenchmark 覆盖
```

## 性能声明

Microbenchmark（BF16，对比 FLA / FlashInfer / CuteDSL；模型-TP cases：Qwen3.6-27B TP1 H16/Hv48、Qwen3.5-397B-A17B TP1 H16/Hv64、TP4 H4/Hv16；uniform 1x8192、1x16384、8x1024、8x2048、16x1024 与 mixed 8@8192/16@8192/16@16384/32@16384；FP32 与 BF16 state）：高 TP/低 V-head 场景 CuteDSL 最优，例如 Qwen3.5-397B-A17B TP4、8x1024、BF16 state 为 FLA 0.793 ms、FlashInfer 0.232 ms（3.43x）、CuteDSL 0.143 ms（5.55x）。E2E speed：Qwen3.6-27B TP1、1xGB200、8k-1k conc256、BF16 state 下 FLA 13977 → CuteDSL 14993 tok/s（+7.3% from FLA）、TPOT 102.55 → 92.17 ms；8k-1 conc256 prefill-only 下 CuteDSL 26350 tok/s（+18.7% over FLA）。Qwen3.5-122B-A10B TEP4、4xGB200、8k-1k conc256、BF16 state 下 FLA 9919 → CuteDSL 12121 tok/s（+22.2% over FLA）、TTFT 1020 → 712.55 ms、TPOT 57.58 → 46.66 ms；FP32 state 为 +11.2%。AIME25（Qwen/Qwen3.6-27B）：Triton/FLA 0.910、FlashInfer 0.908、CuteDSL 0.915。测试条件：硬件 1x/4x GB200、精度 BF16、state dtype FP32/BF16、workload 8k-1k 与 8k-1 conc256。

## 验证与回归风险

正确性测试在 PR 内对比 FLA reference 的 mean/max absolute error，microbench 表列出 FI MAE 与 CuteDSL MAE（量级约 1e-5）；另给出 AIME25 三后端分数与两组 E2E speed。回归风险：kernel 需显式 opt-in（`--gdn-prefill-backend`），默认仍是 FlashInfer；正文 Observations 指出 CuteDSL 在高 TP/低 V heads、skewed requests 明显占优，而长或均匀序列且高 V heads 时 FlashInfer 更优；nsys 显示 TP4 prefill-only 场景 NCCL AllReduce 占 e2e 30%，kernel 改进会被通信噪声淹没，故作者移除了该场景的 benchmark 结果。Future work 列出 KDA 适配、改善 MMA pipeline stall、完整 CUDA Graph 支持、mixed prefill-decode 等未完成项。

## 关联

无（正文仅引用 FLA 设计与 MLSys 2026 FlashInfer Kernel competition 来源，未给出同区域 PR 编号线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
