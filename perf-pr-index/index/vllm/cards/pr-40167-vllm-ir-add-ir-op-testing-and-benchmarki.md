# PR #40167: [vLLM IR] Add IR op testing and benchmarking infrastructure

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gmagogsfm |
| 合并 | 2026-04-21 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/40167 |

## 一句话结论

为 vLLM IR Op 加示例输入生成器与 tolerance 设施，示例 benchmark 显示 geomean 6.48x（84/84 faster）。

## 动机

正文说明为「每个 vLLM IR Op 定义其 example input generator 与 tolerance levels」补充基础设施，从而实现两点：跨所有 provider 的自动精度测试（无需开发者额外投入），以及统一的 benchmarking。本 PR 属于测试/基准设施，而非直接改写某个 kernel。

## 优化点

在 vllm/ir/op.py 的 IrOp 上新增 _input_generator、_tolerance_overrides，以及 has_input_generator、register_input_generator、generate_inputs、override_tolerance、get_tolerance；新增 vllm/ir/tolerances.py 提供 DEFAULT_TOLERANCES 与 ToleranceSpec，get_tolerance 优先用 per-op override、否则查默认表，找不到即报错要求补充。benchmarks/kernels/ir/bench_ir_ops.py 与 shapes.py 提供统一基准，tests/ir/ir_test_utils.py、test_op.py、tests/kernels/ir/test_ir_ops.py、test_layernorm.py 提供自动精度测试。vllm/ir/ops/layernorm.py 为 rms_norm 注册 input generator，并因大 shape 归约误差对 float16 覆盖 tolerance。

## 关键改动

```text
vllm/ir/op.py: IrOp 增加 input generator 与 tolerance 的注册/查询接口
vllm/ir/tolerances.py: 新增 DEFAULT_TOLERANCES 与 ToleranceSpec
vllm/ir/ops/layernorm.py: rms_norm 注册 input generator 并对 float16 覆盖 tolerance
benchmarks/kernels/ir/bench_ir_ops.py、shapes.py: 统一 benchmark 与 shape 集合
tests/ir/ir_test_utils.py、test_op.py、tests/kernels/ir/test_ir_ops.py、test_layernorm.py: 自动精度测试
```

```python
@rms_norm.register_input_generator
def _rms_norm_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-5
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)
    return (x, weight, epsilon)

# Reductions in rms_norm accumulate rounding error at large shapes
rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)
```

## 性能声明

PR 正文给出的是 "Example benchmarking output"（示例输出，用于展示设施），并非本 PR 对某路径的提速声明：vllm_c vs native 为 84/84 faster、0/84 slower，geomean speedup 6.48x，best 15.85x（tokens=2048_hidden=4096_float16），worst 2.62x（tokens=32768_hidden=16384_float32）。测试条件：GPU NVIDIA H100 80GB HBM3，vLLM 0.17.1rc1.dev349+g6792117d7、PyTorch 2.11.0+cu130、CUDA runtime 13.2.78、Triton 3.6.0、CUTLASS 4.4.1、Helion 0.2.4.dev3，Bench mode cuda_graph，Warmup 25 ms（ignored），Repetitions 100 replays，精度按 shape 为 float16 / float32，Git commit 37464b68b。

## 验证与回归风险

新增 tests/ir/ir_test_utils.py、tests/ir/test_op.py、tests/kernels/ir/test_ir_ops.py、test_layernorm.py 与 benchmarks/kernels/ir/bench_ir_ops.py、shapes.py，提供跨 provider 自动精度测试与统一基准。PR 正文未列出具体测试命令、通过数或 CI 结果，也未说明对现有路径的回归影响。从 diff 看新增的是 IrOp 上的元数据注册与测试/基准设施；rms_norm 对 float16 放宽 tolerance（atol=1e-2、rtol=2e-3）以应对 32768×16384 等大 shape 的归约舍入误差，会影响精度判定阈值。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
