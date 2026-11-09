# PR #4218: [Bugfix] Fix marlin kernel crash on H100

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#4218](https://github.com/vllm-project/vllm/pull/4218) |
| 作者 | alexm-redhat |
| 合并日期 | 2024-04-24 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
Marlin 内核的内联 PTX 带 evict_first L2 策略的 async_copy 在 H100 上崩溃,换回标准 async_copy PTX 修复,性能无差异。

## 问题现象
H100 GPU 上使用 Marlin 量化内核时崩溃,该崩溃由 neuralmagic/nm-vllm issue #187 报告(正文以链接形式引用)。影响走 Marlin 量化路径(量化权重 B 的异步拷贝)的推理。

## 根因
正文披露根因:崩溃源于内核中的内联 PTX 汇编——它引入了带 streaming 行为的 async_copy,即使用 `createpolicy.fractional.L2::evict_first` 缓存策略 + `cp.async.cg.shared.global.L2::cache_hint` 的写法(设计意图是量化权重 B 只访问一次、不污染 L2)。该写法在 H100 上导致崩溃;作者明确指出原因是「the inline PTX assembly that introduced the async_copy with streaming behavior」。

## 修复方案
改 `csrc/quantization/marlin/marlin_cuda_kernel.cu`:
1. 把 `cp_async4_stream`(带 `L2::evict_first` fractional L2 policy 的 cache_hint 版本)重命名/替换为标准 `cp_async4`,使用不带 cache hint 的标准 PTX:`cp.async.cg.shared.global [%0], [%1], %2;`。
2. 内核中三处调用点(量化权重 B 的两阶段加载、以及 group_blocks == -1 时写回阶段的 scales 加载)全部由 `cp_async4_stream` 改为 `cp_async4`。
作者说明标准 async_copy PTX 与原写法没有性能差异(原 evict_first 优化失效无损失)。

关键 diff 摘录:
```diff
-__device__ inline void cp_async4_stream(void *smem_ptr, const void *glob_ptr) {
+__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
...
-  asm volatile(
-      "{\n"
-      "   .reg .b64 p;\n"
-      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
-      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
-      "}\n" ::"r"(smem),
+  asm volatile("{\n"
+               "   cp.async.cg.shared.global [%0], [%1], %2;\n"
+               "}\n" :: "r"(smem),
...
-        cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
+        cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;影响 H100 上使用 Marlin 内核的部署。
- 回归面:仅 `marlin_cuda_kernel.cu` 的异步拷贝指令选择。行为语义(global→shared 16B 异步拷贝)不变,仅去掉 L2 evict_first 提示;其他 GPU 架构上同样改用标准 PTX,无 L2 污染保护但作者确认无性能差异。非 Marlin 量化路径不受影响。

## 关联
[neuralmagic/nm-vllm#187](https://github.com/neuralmagic/nm-vllm/issues/187)(正文引用的崩溃报告 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
