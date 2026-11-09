# PR #27010: [HiCache] Fix PP inconsistency with HiCache L3 (#22607)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27010](https://github.com/sgl-project/sglang/pull/27010) |
| 作者 | stepinto |
| 合并日期 | 2026-08-25 |
| labels | high priority, hicache, run-ci, bypass-fastfail, run-ci-extra, release-highlight, unified-radix-cache |
| 热度 | 👍 4 · 💬 52 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

> 注:本卡素材中的 diff 在 40000 字符处截断(文件粒度),截断后的改动未包含在素材中,未据此编造。

## 一句话结论

HiCache 与 PP 同用时各 PP rank 对 ack/prefetch 队列消费数量与 L3 命中长度不一致,导致 rank diverge 直至 crash;引入 pp_sync 队列计数同步与 prefetch 线程 all_reduce(MIN) 修复。

## 问题现象

同时部署 HiCache 与 PP 时,多个 PP rank 之间发生 divergence,并最终导致 sglang crash。正文的详细原因讨论指向 issue #22607(素材未包含该 issue 内容)。影响面为 HiCache(hierarchical cache,含 L3 storage prefetch)与 pipeline parallelism 组合部署的场景;正文未描述具体报错日志或最小复现。

## 根因

正文在 Modifications 中披露了两类不一致(详细原因在 issue #22607 中讨论,素材未含其内容):

1. 队列消费数量不一致:PP0 的 scheduler 线程决定其本地队列(ack_load_queue、ack_prefetch_queue 等)中 completion event 的数量后,PP1 必须从自己的队列消费**完全相同数量**的事件;此前没有这种约束,rank 间消费进度错开即造成 divergence。
2. prefetch 结果长度不一致:PP0 与 PP1 对同一序列在各自 L3 storage 上的长度可能不同,必须通过 all_reduce(MIN) 就长度达成一致(与 TP 的处理一致)。旧 `check_prefetch_progress` 直接在调用处 all_reduce(MIN),但 PP 下无法在 scheduler 线程中执行 collective,因此该同步在 PP 场景失效。

## 修复方案

引入两种同步机制:

- **pp_sync(scheduler 线程之间)**:新增 `PrefetchAck` 与 `ack_prefetch_queue`;`_drain_queue` 改为按给定数量 `n` 消费(不足时阻塞),`n` 取自跨 rank all_reduce(MIN) 后的各队列 `qsize`,保证所有 TP/PP rank 消费相同数量的事件;`_sync_hicache_ready_counts` 与 `drain_storage_control_queues` 将 ack_prefetch_queue 纳入统计。`_page_transfer` 在出错/终止时也保持产出与其他 rank 相同数量的 PrefetchAck,以免 reduce 侧错位(代码注释称「This is so tricky」)。
- **all_reduce(prefetch 线程之间)**:新增后台 `prefetch_sync_thread`,在 `_reduce_prefetch_ack` 中对 `completed_tokens` 做 all_reduce(MIN),使各 rank 就 prefetch 前缀长度达成一致;`can_terminate_prefetch` 简化,终止决策由 PP rank 0 给出再 all_reduce(MAX);`_clamp_prefetch_result` 不再自行跨 rank 同步(ack 已同步),并新增 `_page_transfer_kv_batch` 将 KV 与 sidecar pool 命中数 clamp 到最小;同时移除 `_barrier_attn_groups` 与 queued-limit abort 路径上的 `terminate_prefetch`。

关键摘录(hiradix_cache.py):

```diff
         def _drain_queue(q: queue.Queue, n: Optional[int]):
-            # 旧实现:limit 内 get_nowait,Empty 即停
+            if n is None:
+                while not q.empty():
+                    item = q.get()
+                    yield item
+            else:
+                for _ in range(n):
+                    # Block when there is no enough elements.
+                    # All TP/PP ranks must consume the same number of elements.
+                    item = q.get()
+                    yield item
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围,不写。
- 回归面:按 diff,HiCache 的 storage 控制流被重构——`cache_controller.py` 的 PrefetchOperation/ack 队列与线程模型(新增 prefetch_io_aux_thread、prefetch_sync_thread)、`hiradix_cache.py` 的 prefetch 进度检查/终止/结果插入路径、`hicache_storage.py` 的 pool hit 统计、`scheduler.py` 的 abort 路径。所有启用 hierarchical cache(尤其 L3 storage backend)及 PP/TP/CP 组合的部署均走新同步逻辑;线程与阻塞式 get 增加了调度路径上的新时序依赖。验证:新增 `test_unified_radix_cache_hicache_pp_kl.py`,SWE-bench Verified(PP=4,TP=2,L3)353/500 vs 基线 356/500,差异 7 例均归因于容器超时退出。

## 关联

- issue [#22607](https://github.com/sgl-project/sglang/issues/22607)(正文指明的详细原因讨论处)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
