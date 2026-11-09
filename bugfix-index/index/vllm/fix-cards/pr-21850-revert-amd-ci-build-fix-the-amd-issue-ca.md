# PR #21850: Revert "[AMD][CI/Build] Fix the AMD issue caused by inappropriate of symbol exposure (#21647)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#21850](https://github.com/vllm-project/vllm/pull/21850) |
| 作者 | gshtras |
| 合并日期 | 2025-07-29 |
| labels | rocm, ready, ci/build |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#21647 把 int8_quant_kernels.cu 挪出通用源码列表导致 ROCm 构建损坏;回滚之,正确修复已由 #21766 完成。

## 问题现象
正文披露:#21647 的改动是 breaking 的——ROCm 上仍需要编译 `int8_quant_kernels.cu` 文件的其余部分,该 PR 却把它从通用编译列表移走,导致 AMD/ROCm 构建出问题。

## 根因
正文披露:#21647 为解决「symbol exposure 不当」问题,把 `csrc/quantization/compressed_tensors/int8_quant_kernels.cu` 移入 CUDA 专属源码列表;但该文件的其余部分在 ROCm 上也必须编译,因此该改动破坏了 ROCm 构建。作者指出正确做法已在 #21766 落地,故直接回滚 #21647。

## 修复方案
回滚 `CMakeLists.txt` 中 #21647 的源码列表调整:把 `int8_quant_kernels.cu` 重新加回通用 `VLLM_EXT_SRC` 列表(ROCm 也会编译),并从 `VLLM_GPU_LANG STREQUAL "CUDA"` 专属列表中移除:

```diff
   "csrc/quantization/gptq/q_gemm.cu"
+  "csrc/quantization/compressed_tensors/int8_quant_kernels.cu"
   "csrc/quantization/fp8/common.cu"
...
-    "csrc/quantization/fp8/per_token_group_quant.cu"
-    "csrc/quantization/compressed_tensors/int8_quant_kernels.cu")
+    "csrc/quantization/fp8/per_token_group_quant.cu")
```

## 影响与回归风险
- 影响窗口:#21647 合入至本回滚之间的 ROCm 构建受影响;正文未给出具体版本号。
- 回归面:仅 CMake 源码组织;`int8_quant_kernels.cu` 恢复为全平台编译,#21647 原本针对的符号暴露问题转由 #21766 处理,CUDA 构建产物内容不变(同一文件仍参与编译,只是归属列表变化)。

## 关联
- #21647(被回滚的 PR)
- #21766(正文指认的正确修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
