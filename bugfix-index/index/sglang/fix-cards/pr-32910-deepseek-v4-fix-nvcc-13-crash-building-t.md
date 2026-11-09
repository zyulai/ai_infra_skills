# PR #32910: [DeepSeek-V4] Fix nvcc 13 crash building the topk_v2 kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32910](https://github.com/sgl-project/sglang/pull/32910) |
| 作者 | guptaishaan |
| 合并日期 | 2026-08-03 |
| labels | GLM, blackwell, bug, deepseek, hopper, jit-kernel, kernel, nvidia, run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论

CUDA 13 的 cicc 编译器对「cluster 映射 DSMEM 指针写入 problem.out 再被加载」的模式段错误,致 topk_v2 JIT 构建失败、DeepSeek-V4 服务启动即死;把映射别名隔离进传值副本修复。

## 问题现象

Hopper 或 Blackwell 主机 + CUDA 13 上,所有 DeepSeek-V4 服务在 CUDA graph capture 阶段死亡,报 `ninja exited with status 139`——`dpsk_v4_topk_v2` JIT 模块整体构建失败(cicc 段错误)。

## 根因

正文披露:`topk_small_batch_kernel` 把 cluster 映射的 DSMEM 地址(`cluster.map_shared_rank(topk_indices, worker_rank)`)放进 `problem.out`,而 `problem_transform` epilogue 随后会经同一指针加载;在 CUDA 13.x 上 cicc 对这次合并发生段错误。作者在 sglang 之外用报告者的工具链(nvcc 13.2.86 + gcc 13.4,sm_90a)复现:同样两条 nvcc warning、同样 `Segmentation fault (core dumped)`、exit 139;二分定位到「DSMEM 指针 + 后续加载」的组合——单独去掉任一即可不崩,`-Xcicc -O1` 能编译未修补文件,确认是 cicc 优化器崩溃而非源码错误。

## 修复方案

`python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` 的 `topk_small_batch_kernel`:映射别名只放进副本 `peer_problem`,仅传给按值收参的 `Cluster::forward`;当选 rank(`blockIdx.y == worker_rank`)仍通过自己的 `topk_indices` 读回同样的字节,与相邻的 Register4 / Streaming 分支一致,无行为变化。

关键 diff 摘录:

```diff
-    problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
-    Cluster::forward<kPDL>(problem, &smem);  // write to peer's output shared memory
+    auto peer_problem = problem;
+    peer_problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
+    Cluster::forward<kPDL>(peer_problem, &smem);  // write to peer's output shared memory
     cluster.sync();
```

验证:修补后同一编译单元可用 nvcc 13.2.86 编译,且 nvcc 12.6 仍可编译;新增 `test_topk_v2_compiles_for_sm90a`(对 `load_jit()` 生成的编译单元跑 `nvcc -ptx -arch=sm_90a`,修复前 exit 139、修复后通过;无需 cluster GPU,无 nvcc 时跳过)。

## 影响与回归风险

- 影响窗口:CUDA 13.x + Hopper/Blackwell 上的 DeepSeek-V4 服务启动路径;具体版本号正文未说明,不写。
- 回归面:仅 `topk_small_batch_kernel` 非 elected 分支的局部变量结构,数据写入目标不变(仍写 peer 的输出共享内存)。作者明示未在 GPU 上跑过:`topk_v2.cuh` 在其可用的 A40(sm_86)上根本不编译(`__cluster_dims__ is not supported`),`test_topk_v2.py` 既有 244 个正确性用例修复前后都没跑,需在 Hopper/Blackwell 上补跑;JIT 树其余部分未对 CUDA 13.2 编译过,启动后段可能还有其他 cicc 崩溃(其他 JIT 核未用 `map_shared_rank`,无此特定触发)。

## 关联

- [#32830](https://github.com/sgl-project/sglang/issues/32830)(正文注明 Fixes #32830,含报告者的完整启动日志)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
