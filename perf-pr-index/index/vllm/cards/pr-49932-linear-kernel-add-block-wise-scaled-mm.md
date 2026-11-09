# PR #49932: [Linear] [Kernel] add block-wise scaled_mm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zufangzhu |
| 合并 | 2026-08-06 |
| 分类 | Quantization(major) |
| Label | (none) |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/49932 |

## 一句话结论

为 DeepSeek 式 block-wise FP8 GEMM 新增无外部依赖的 torch._scaled_mm 原生后端,compute-bound 时比次优后端快至 1.09×。

## 动机

DeepSeek 风格的 FP8 block scaling(1x128 激活 + 128x128 权重)此前依赖 Triton / CUTLASS / DeepGEMM 后端,它们要么需要额外库,要么需要 per-shape tuning。PR 补充一条原生 PyTorch 路径:零依赖、无需调参,作为既有后端之外的备选。

## 优化点

新增 `BlockWiseTorchFP8ScaledMMLinearKernel`,继承 `Fp8BlockScaledMMLinearKernel`,把 (1x128, 128x128) 的 block scale 布局路由到 `torch._scaled_mm` 的 block path(该 op 按 scale 张量形状派发,要求 fp32 scale、`scale_a=[M, ceil(K/128)]`、`scale_b=[ceil(K/128), ceil(N/128)]`)。CUDA 上仅限 SM90(Hopper),与 PyTorch `_check_deepseek_support()` 的实现范围一致。cuBLASLt 要求 M 为 4 的倍数,故对 A 及其 column-major scale 做 M 方向 padding 后再 slice 回结果(padded 行不初始化,不影响输出)。激活量化走 `QuantFP8`,CUDA 上使用 column-major scales。

## 关键改动

```text
vllm/model_executor/kernels/linear/scaled_mm/pytorch.py: 新增 kernel 类与 apply_block_scaled_mm(M padding)
vllm/model_executor/kernels/linear/__init__.py: 注册进 pytorch backend 与 CUDA fallback 列表
tests/kernels/quantization/test_block_fp8.py: SM90-gated 端到端数值测试(含 M=83 padding 路径)
```

## 性能声明

PR 正文仅一句:"stays competitive-to-best in the compute-bound regime (M≥1024, up to 1.09x faster than the next-best backend on H20)"。即 H20 上 M≥1024 时最高比次优后端快 1.09×;未给出具体 shape/表格数据,小 M 表现未注明。

## 验证与回归风险

新增 `test_w8a8_block_fp8_torch_scaled_mm_matmul`(SM90-gated),M=83/N=576/K=7168 覆盖 M-padding 路径,与 `native_w8a8_block_matmul` 参考实现比对 `rel_diff < 0.001`。回归风险:仅 SM90 支持(其他 arch 报不支持);要求精确的 1x128 激活 + 128x128 权重 group shape,否则 `can_implement` 拒绝;pad 分支只在 `current_platform.is_cuda()` 生效(XPU/ROCm 走原样);依赖 PyTorch `_scaled_mm` block path 的行为,PyTorch 侧语义变化会直接影响本后端。

## 关联

同区域为 FP8 block-scale GEMM 后端选型(既有 Triton/CUTLASS/DeepGEMM 后端);pack 正文未列出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
