# PR #32952: [Refactor] Rename `gptq_marlin` to `marlin` to match MoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-01-23 |
| 分类 | MoE |
| Label | performance, ready, ci/build |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 为目录/符号重命名,大量文件 similarity 100% |
| 链接 | https://github.com/vllm-project/vllm/pull/32952 |

## 一句话结论

纯重命名重构:把 dense 侧 `gptq_marlin` 目录与 `gptq_marlin_gemm` op 统一改名为 `marlin`,与 MoE 侧 `marlin_moe_wna16` 命名对齐,不改变任何 kernel 行为。

## 动机

dense Marlin GEMM 历史上以 `gptq_marlin` 命名(`csrc/quantization/gptq_marlin/`、`ops.gptq_marlin_gemm`),但该 kernel 实际早已不止支持 GPTQ——torch binding 注释更新后写明它支持 GPTQ、AWQ、FP8、NVFP4、MXFP4;同时 MoE 侧对应实现叫 `marlin_moe_wna16`。两套命名不一致容易误导。此 PR 消除命名漂移,无性能改动。

## 优化点

机制层面无优化,全部为等价重命名:`csrc/quantization/gptq_marlin/` 目录整体更名为 `csrc/quantization/marlin/`,`gptq_marlin.cu` 更名为 `marlin.cu`,C++ 函数与 TORCH library 注册名 `gptq_marlin_gemm` 更名为 `marlin_gemm`(`TORCH_LIBRARY_IMPL_EXPAND` 与 `csrc/torch_bindings.cpp` 中的 `ops.def` 同步);Python 侧 `vllm/_custom_ops.py` 的 `gptq_marlin_gemm` 包装函数与 `_C::gptq_marlin_gemm` fake 注册同步改名;`marlin_utils.py`/`marlin_utils_fp4.py`/`marlin_utils_fp8.py` 的调用点以及 CMakeLists 的源文件路径、MoE/allspark 头文件 include 路径全部更新。`gptq_marlin_repack`、`awq_marlin_repack`、`gptq_marlin_24_gemm` 等格式专属名字保留。

## 关键改动

```text
csrc/quantization/gptq_marlin/ → csrc/quantization/marlin/: 目录整体改名(多数文件 100% 相似)
csrc/quantization/marlin/marlin.cu: gptq_marlin_gemm 函数与 TORCH 注册改名 marlin_gemm
csrc/torch_bindings.cpp: op schema def 改名,注释更新为支持 GPTQ/AWQ/FP8/NVFP4/MXFP4
vllm/_custom_ops.py: Python 包装函数与 register_fake 改名
vllm/model_executor/layers/quantization/utils/marlin_utils*.py: 所有调用点更新
```

注册改名:

```cpp
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("marlin_gemm", &marlin_gemm);
}
```

## 性能声明

无。PR 正文未给出任何 benchmark 数字(勾选清单的 test plan/test result 亦为空)。重命名不涉及任何计算路径变化。

## 验证与回归风险

依赖测试 `tests/kernels/quantization/test_marlin_gemm.py`(测试函数 `test_gptq_marlin_gemm` 同步改名 `test_marlin_gemm`)与 benchmark 脚本 `benchmarks/kernels/benchmark_marlin.py`、`benchmark_machete.py` 的调用更新。风险面主要是构建与外部引用:CMake 源路径 glob 更改,若下游脚本直接引用 `ops.gptq_marlin_gemm` 或 `_C::gptq_marlin_gemm` 会失败;ROCm 上这些 op 本就不支持(测试模块级 skip 说明同步更新)。对运行时数值与性能无影响。

## 关联

同区域的 Marlin 系列实现:`csrc/moe/marlin_moe_wna16/`(命名对齐的目标)、`gptq_marlin_repack`/`awq_marlin_repack` repack kernel、`marlin_24` sparse kernel。pack 未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
