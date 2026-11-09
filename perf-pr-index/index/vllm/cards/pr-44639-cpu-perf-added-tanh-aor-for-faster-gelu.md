# PR #44639: [CPU][Perf]Added tanh AOR for faster gelu activations.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @almayne |
| 合并 | 2026-07-01 |
| 分类 | Other（tier: major） |
| Label | ci/build, cpu, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/44639 |

## 一句话结论

CPU SIMD 上新增 tanh AOR 加速 gelu tanh 近似，1 线程下 fp32 最快 1.860x、bf16 最快 2.588x。

## 动机

为 SIMD 架构添加 Arm Optimised Routine（AOR）的 tanh 实现，并在使用 tanh 近似的 gelu activations 上启用，以加速 CPU 侧 gelu。对比基线是 PyTorch 的 `F.gelu(t, approximate="tanh")`。

## 优化点

新增 Arm SIMD / NEON 的 tanh 优化例程，实现并注册为 CPU 自定义 op `torch.ops._C.gelu_tanh`（及 `gelu_tanh_and_mul`），在 CPU 平台的 gelu tanh 近似路径启用（在 `vllm/platforms/cpu.py` 中把 `+gelu_and_mul` 加入 `custom_ops`）。实现涉及 `csrc/cpu/cpu_tanhf_neon.hpp`、`activation.cpp`、`cpu_types_arm.hpp`、`torch_bindings.cpp`、`ops.h` 与 `cmake/cpu_extension.cmake`（ASIMD 编译分支加入新源文件）。benchmark script 以 `torch.testing.assert_close` 对 fp32 / bf16 两种 dtype 做数值校验。

## 关键改动

```text
csrc/cpu/cpu_tanhf_neon.hpp: 新增 Arm SIMD/NEON 的 tanh 优化例程（AOR）
csrc/cpu/activation.cpp、csrc/cpu/torch_bindings.cpp、csrc/ops.h: 实现并注册 gelu_tanh / gelu_tanh_and_mul CPU 自定义 op
cmake/cpu_extension.cmake、csrc/cpu/cpu_types_arm.hpp: 在 ASIMD 编译分支加入新源文件与类型支持
vllm/model_executor/layers/activation.py: 把 gelu tanh 近似路径接到新 CPU op
vllm/platforms/cpu.py: 在 CPU 平台把 +gelu_and_mul 加入 custom_ops
tests/kernels/core/test_cpu_activation.py: 新增 test_cpu_gelu_tanh_and_mul 单测
```

## 性能声明

测试条件：CPU（Arm SIMD / ASIMD），benchmark script 对比 `F.gelu(t, approximate="tanh")` 与 `torch.ops._C.gelu_tanh`，shape (4096,4096) / (8192,4096) / (16384,4096)，dtype float32 与 bfloat16，`num threads` 1 与 8（由 `OMP_NUM_THREADS` 控制），iters=200、warmup=50，`torch.testing.assert_close`（bf16 rtol/atol 1e-2，fp32 rtol 1e-4 / atol 1e-5）。相对 PyTorch `F.gelu(tanh)` 的 speedup：1 线程 float32 为 1.860x / 1.858x / 1.857x、bfloat16 为 2.588x / 2.586x / 2.587x；8 线程 float32 为 1.681x / 1.676x / 1.667x、bfloat16 为 2.378x / 2.373x / 2.372x。PR 正文未注明具体 CPU 型号，仅说明 SIMD / Arm ASIMD 架构。

## 验证与回归风险

单测：`python -m pytest -x -v -s tests/kernels/core/test_cpu_activation.py::test_cpu_gelu_tanh_and_mul` 通过；benchmark script 内用 `torch.testing.assert_close` 对两种 dtype 做数值校验（bf16 rtol/atol 1e-2，fp32 rtol 1e-4 / atol 1e-5）。回归风险：仅 Arm SIMD / ASIMD CPU 路径（cmake 中 `ASIMD_FOUND AND NOT APPLE_SILICON_FOUND` 分支）且用于 tanh 近似的 gelu；数值精度按上述容差验证；正文未给出其他受影响模型 / 硬件信息。

## 关联

无（PR 正文未给出其他 PR 编号线索；`Co-authored-by: GPT 5.4` 非 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
