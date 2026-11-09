# PR #21393: [Bugfix][ROCm][Build] Fix build regression on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#21393](https://github.com/vllm-project/vllm/pull/21393) |
| 作者 | gshtras |
| 合并日期 | 2025-07-23 |
| labels | rocm, ready, ci/build |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#21083 把 CUDA 专属的 per_token_group_quant 移出 ifdef 编译段致 ROCm 构建回归,将其移回 CUDA-only 区域修复。

## 问题现象
正文未描述具体现象,仅说明是 ROCm 上的构建回归(build regression),并定位为 #21083 引入的问题,修法是把 CUDA 专属文件移回 ifdef CUDA 段。

## 根因
正文披露:回归来自 #21083,需要「Moving CUDA specific file to the ifdef CUDA section」。由 diff 推断:#21083 将 `csrc/quantization/fp8/per_token_group_quant.cu` 源文件、`per_token_group_quant_fp8` 的 ops 声明与 torch binding 放到了无条件编译路径,ROCm 构建时也会编译/注册这个 CUDA 专属算子,导致构建失败。

## 修复方案
三处把 CUDA 专属内容移回 CUDA-only 区域:
1. `CMakeLists.txt`:将 `"csrc/quantization/fp8/per_token_group_quant.cu"` 从通用 `VLLM_EXT_SRC` 列表移入 `VLLM_GPU_LANG STREQUAL "CUDA"` 分支;
2. `csrc/ops.h`:`per_token_group_quant_fp8` 声明移入 CUDA 守卫 `#endif` 之前;
3. `csrc/torch_bindings.cpp`:`ops.def`/`ops.impl("per_token_group_fp8_quant", ...)` 移入 `#ifndef USE_ROCM` 块。

## 影响与回归风险
- 影响窗口:正文说明自 #21083 起受影响。
- 回归面:ROCm 与 CUDA 的构建文件及 binding 注册;CUDA 侧编译内容不变(仅位置移动),非 CUDA/ROCm 平台不再暴露该 binding,这正是恢复原状。

## 关联
[#21083](https://github.com/vllm-project/vllm/pull/21083)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
