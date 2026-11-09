# PR #50268: [Hardware][AMD] Enable fused bf16→fp32 router GEMM on ROCm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mpashkovskii |
| 合并 | 2026-08-12 |
| 分类 | Attention / Kernel \| tier: major |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/50268 |

## 一句话结论

ROCm 把 MoE router gate 的 fp32 cast 折进 GEMM epilogue，gfx942 上 TPOT/ITL/输出吞吐均约改善 1.6%。

## 动机

MoE router gate 需要 `out_dtype=torch.float32` 供 `grouped_topk` 使用。正文指出 ROCm 上 fused fp32-output GEMM 的实现此前只对 CUDA 平台开放，导致回退到"bf16 GEMM + 单独 cast"，在每个 MoE layer、每个 decode step 的 router GEMM 与 `grouped_topk` 之间都出现一个独立的 `bf16→fp32 copy kernel`。本 PR 用于解决 issue #50267。

## 优化点

通过修改 `vllm/model_executor/layers/fused_moe/router/gate_linear.py` 启用 hipBLASLt 的 `out_dtype` epilogue 支持（类比 CUDA 的 cuBLAS 能力）：在"无 bias、权重为 bf16、输出为 fp32"时把 ROCm 作为 `allow_cublas_router_gemm` 的另一条成立条件，并同时更新 `__init__` 与 `set_out_dtype` 两个方法以保持一致。fused tier 由此在 ROCm 上执行，把 fp32 cast 折进 GEMM epilogue，去掉每个 step 的转换 kernel。CUDA 不受影响（条件用 `current_platform.is_rocm()`，在 CUDA 上为 false）；改动范围仅 router gate，expert GEMM 及其他 fp32 操作不变。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/router/gate_linear.py: __init__ 与 set_out_dtype 增加 ROCm 分支，启用 fp32 out_dtype epilogue 的 fused router GEMM
tests/kernels/test_gate_linear_rocm_dispatch.py: 新增 device-free 分派测试（mock 平台谓词），覆盖 bias / 权重 dtype / out_dtype / 平台条件
```

## 性能声明

Benchmarked on ROCm (gfx942) 的 online serving benchmark，baseline 对 fused、各 3 个 seed（123 / 456 / 789），max concurrency 4，每次 40 requests（40960 input + 40960 generated tokens），全部 40/40 成功、0 失败，取 3 seed 平均：Benchmark duration 343.18 → 337.69 s（−5.5 s，−1.6%）；Output throughput 119.35 → 121.29 tok/s（+1.9，+1.6%）；Total throughput 238.71 → 242.59 tok/s（+3.9，+1.6%）；Mean TPOT 32.87 → 32.34 ms（−0.53，−1.6%）；Mean ITL 32.87 → 32.34 ms（−0.53，−1.6%）；Mean TTFT 689.65 → 688.97 ms（≈flat，−0.1%）。正文称 decode 路径指标一致改善约 1.6%、TTFT 基本不变，seed 间方差小（throughput ±0.2 tok/s，TPOT ±0.07 ms）。PR 未注明模型与精度（dtype）条件。

## 验证与回归风险

Correctness 在 gfx942 上验证，结果与 fp32 reference 在 ~6e-5 tolerance 内一致。Testing Plan：确认 ROCm traces 中不再出现独立的 `bfloat16tofloat32_copy_kernel`、校验 router logits 与原行为在容差内一致、并新增单元测试。新增 `tests/kernels/test_gate_linear_rocm_dispatch.py` 以 device-free 方式 mock 平台谓词，覆盖无 bias/bf16/fp32 启用、有 bias、fp32 权重、非 fp32 out_dtype、非 ROCm 非 CUDA 等分派分支。正文未给出测试通过计数，也未列出其他回归风险。

## 关联

正文明确提到 issue #50267（本 PR 要解决的问题）。此外未给出同区域其他 PR 编号线索，无其他关联。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
