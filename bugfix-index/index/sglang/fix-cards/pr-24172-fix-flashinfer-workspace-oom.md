# PR #24172: Fix flashinfer workspace OOM

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24172](https://github.com/sgl-project/sglang/pull/24172) |
| 作者 | kpham-sgl |
| 合并日期 | 2026-05-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
FlashInfer 创建 allreduce fusion workspace 时 OOM,单个 rank 失败退出而 peers 等待 handle 交换,NCCL watchdog 超时挂死;改为创建前用真实 CUDA driver 分配序列做 preflight 探测,并在 CPU 组上全员投票,失败则集体跳过 fusion。

## 问题现象
正文披露:`test/registered/8-gpu-models/test_gpt_oss_120b.py` 有时一直挂起直到 NCCL watchdog timeout,原因是 FlashInfer create workspace OOM(附 CI 失败链接 run 24918423411/job 72975439419)。影响面是启用 FlashInfer allreduce fusion 的多卡推理,显存紧张时偶发。

## 根因
正文披露:此前判断「显存是否够」用的是 `torch.cuda.memory.get_mem_info`,探测的内存量与类型都不对,导致明明放不下 workspace 却仍进入 `create_allreduce_fusion_workspace`,在创建中 OOM。机制(diff 中新增 `_preflight_check_workspace_memory` 的 docstring 原文):FlashInfer TRTLLM workspace 会分配多个 SymmDeviceMemory 缓冲并跨 rank 交换 handle,「If one rank fails local cuMemCreate and exits while peers enter handle exchange, peers can hang until the watchdog aborts」——单 rank 失败退出、peers 卡在集合通信,直到 watchdog 中止。

## 修复方案
1. `python/sglang/srt/layers/flashinfer_comm_fusion.py` 新增 preflight:`_make_flashinfer_workspace_allocation_prop` 构造与实际创建一致的 CUmemAllocationProp(POSIX FD 或 FABRIC handle 类型);`_flashinfer_trtllm_workspace_allocation_sizes` 镜像 FlashInfer TRTLLM SymmDeviceMemory 的本地分配尺寸(buffer/flag/3×lamport,各按 2MiB 对齐,再叠加 cuMemGetAllocationGranularity 与 multicast granularity);`_probe_cumem_create_sequence` 用真实 `cuMemCreate` 逐个试分配、finally 中 `cuMemRelease` 归还;
2. `_preflight_check_workspace_memory`:本地探测后在 CPU group 上 `dist.all_reduce(flag, op=ReduceOp.BAND)` 投票,任一 rank 失败则所有 rank 一致返回 False;`FlashInferWorkspaceManager.initialize` 在创建 workspace 前先 preflight,失败则置 `_flashinfer_allreduce_unavailable = True` 并跳过,不再进入创建;
3. `python/sglang/srt/utils/common.py` 新增 `get_cuda_driver_bindings()`(兼容 `cuda.bindings.driver` 与旧 `cuda.cuda` 导入);
4. 新增 `test/registered/distributed/test_flashinfer_fusion_preflight.py`:两卡分布式测试,「rank0_starved」场景让 rank 0 用相同 allocation prop 预留几乎全部空闲显存,断言 rank 0 本地探测失败、BAND 广播后两个 rank 均返回 False;另有正常路径投票 PROCEED 的用例。

关键 diff 摘录:

```diff
+        global _flashinfer_allreduce_unavailable
+        if not _preflight_check_workspace_memory(
+            world_size=world_size,
+            max_token_num=max_token_num,
+            hidden_dim=hidden_dim,
+            dtype=dtype,
+            cpu_group=cpu_group,
+        ):
+            _flashinfer_allreduce_unavailable = True
+            self.workspace = None
+            self.initialized = False
+            return
```

## 影响与回归风险
- 回归面:`FlashInferWorkspaceManager.initialize` 在所有走 FlashInfer allreduce fusion 的多卡部署上都会先做一次 preflight(每个分配一次 cuMemCreate/cuMemRelease 试探,有小额延迟开销);探测失败或投票不通过时功能从「崩溃/挂死」变为「跳过 fusion、回退普通 allreduce」;`global` 声明位置从 except 分支上移。依赖 `cuda.bindings` 驱动绑定可用,不可用时落入 except 分支按探测失败处理(跳过 fusion)。

## 关联
- https://github.com/sgl-project/sglang/actions/runs/24918423411/job/72975439419 (CI 失败 run)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
