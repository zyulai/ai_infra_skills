# PR #25843: Update launch_bounds_utils.h for correct compile on Multiple Cuda Arch - PTXAS out of range Warning

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25843](https://github.com/vllm-project/vllm/pull/25843) |
| 作者 | DrStone1971 |
| 合并日期 | 2025-10-01 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 28 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论
`launch_bounds_utils.h` 给 `__launch_bounds__` 的每 SM 线程值未按架构区分,SM120(Nvidia 5090)编译时 ptxas 报 out of range 并忽略;改为按架构族映射。

## 问题现象
在 Nvidia 5090(SM120 架构)上编译 vLLM CUDA 代码时,大量 kernel 触发 ptxas 警告:
```
ptxas warning : Value of threads per SM for entry _ZN4vllm15cvt_fp16_to_fp4I13__nv_bfloat16Lb0EEEviiPKT_PKfPjS7_ is out of range. .minnctapersm will be ignored
```
波及 `nvfp4_quant_kernels.cu`、`activation_nvfp4_quant_fusion_kernels.cu`、`nvfp4_experts_quant.cu`、`moe_permute_unpermute_kernel.cu` 等编译单元(编译日志第 60-63/477 步)。

## 根因
正文披露:`launch_bounds_utils.h` 需要为 `__launch_bounds__` 传入正确的值以支持 SM120 架构,该头文件是编译期估算每 SM 最大线程数的地方。(由 diff 推断)旧逻辑只有 `__CUDA_ARCH__ < 300` 时取 1536,其余架构(含 SM120)一律取 2048;而 sm_120 硬件上限是 1536 线程/SM,传给 `__launch_bounds__` 的值超出范围后 ptxas 直接忽略 `.minnctapersm`,发出上述警告(理论上也可能影响占用率/调度假设)。

## 修复方案
重写 `csrc/launch_bounds_utils.h` 中 `VLLM_MAX_THREADS_PER_SM` 的编译期映射,按架构族区分:1024 线程/SM 给 Turing(sm_75);1536 给 Ampere GA10x(sm_86/87)、Ada(sm_89)、GB20x 消费卡(sm_120/121)、Thor(sm_101/sm_110);2048 给 Volta(sm_70/72)、Ampere GA100(sm_80)、Hopper(sm_90)、Blackwell(sm_100/103);未知未来架构与宿主编译过程(无 `__CUDA_ARCH__`)回退 2048。

```diff
+    /* 1536 thr/SM: Ampere GA10x (sm_86/87), Ada (sm_89),
+        GB20x consumer (sm_120/121), Thor (sm_101 or sm_110) */
+    #elif (__CUDA_ARCH__ == 860) || (__CUDA_ARCH__ == 870) || \
+        (__CUDA_ARCH__ == 890) || (__CUDA_ARCH__ == 1010) ||  \
+        (__CUDA_ARCH__ == 1100) || (__CUDA_ARCH__ == 1200) || \
+        (__CUDA_ARCH__ == 1210)
+      #define VLLM_MAX_THREADS_PER_SM 1536
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:所有使用 `VLLM_MAX_THREADS_PER_SM` 计算 `__launch_bounds__` 的 CUDA kernel。按 diff,多个架构的取值被修正性下调:sm_75 由 2048 变 1024,sm_86/87/89/120/121 等 2048 变 1536;这些值只影响启动边界的编译期假设,越界告警消除(作者附 compile_ptxas.log 验证),kernel 语义不变;Host pass 与未知架构保持 2048 默认。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
