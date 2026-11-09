# PR #21384: Revert "[Refactor] Fix Compile Warning #1444-D (#21208)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#21384](https://github.com/vllm-project/vllm/pull/21384) |
| 作者 | yewentao256 |
| 合并日期 | 2025-07-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | regression |

## 一句话结论
#21208 用 cuda::std::plus 替换 cub::Sum 疑似弄坏 AMD 编译;先回退恢复编译,警告另开 PR 修。

## 问题现象
AMD 设备上编译失败。正文原话:"Seems this break the compilation on AMD device, let's revert this first... so that we don't block users"。正文未附具体报错日志。

## 根因
正文:疑似被回退的 #21208 破坏了 AMD 设备上的编译("Seems this break the compilation on AMD device"),具体机制未披露。从 diff 看,#21208 曾把 `csrc/moe/topk_softmax_kernels.cu` 中的 `cub::Sum sum;` 换成 `cuda::std::plus<float> sum;` 并引入 `#include <cuda/std/functional>` 头文件(由 diff 推断)。

## 修复方案
整体回退 commit 6e5b5ca580e60a3b7a6c5613ae7d8676f2c8cbc6:`csrc/moe/topk_softmax_kernels.cu` 恢复 `cub::Sum sum;`,移除 `#include <cuda/std/functional>`。作者表示之后会另开 PR 修复该编译警告并同时支持 AMD(需要时间搭 AMD 服务器环境)。

```diff
-#include <cuda/std/functional>
@@ -63,7 +62,7 @@ __launch_bounds__(TPB) __global__
 
     const int thread_row_offset = blockIdx.x * num_cols;
 
-    cuda::std::plus<float> sum;
+    cub::Sum sum;
```

## 影响与回归风险
- 影响窗口:正文未写(自 #21208 合入起,本 PR 为其回退)。
- 回归面:MoE topk_softmax CUDA 内核求和实现恢复为 `cub::Sum`;#21208 原本要修的编译警告 #1444-D 将重新出现(作者已说明会另开 PR 处理)。

## 关联
- #21208(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
