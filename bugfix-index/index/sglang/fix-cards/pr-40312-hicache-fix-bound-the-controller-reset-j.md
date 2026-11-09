# PR #40312: [HiCache] fix: bound the controller reset join so a stalled storage thread cannot hang the scheduler

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#40312](https://github.com/sgl-project/sglang/pull/40312) |
| 作者 | alphabetc1 |
| 合并日期 | 2026-09-23 |
| labels | hicache, run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
HiCacheController.reset() 对存储线程不限时 join,线程卡住会挂死 scheduler;改用带超时与存活检查的 _stop_storage_threads。

## 问题现象
正文描述:`HiCacheController.reset()` 先置 `storage_stop_event`,随后对 4 个存储线程做不限时 `join()`;若某线程不退出,reset 会无限等待并阻塞 scheduler。作者注明一种可能的停滞场景(由源码推断、未复现):prefetch 线程阻塞在跨 rank 集合通信 `all_reduce()` 上;并明确本 PR 只限定 reset 的等待时间,不解决集合通信停滞本身,定位为 hardening 加固。

## 根因
正文披露:reset() 使用 untimed `join()`,对停滞线程会无限期等待,从而阻塞调度器;且旧逻辑即使线程仍存活也会继续清空共享状态、重启存储线程。

## 修复方案
`python/sglang/srt/managers/cache_controller.py` 的 `reset()`:删除 `storage_stop_event.set()` 与对 prefetch / prefetch_io_aux / prefetch_sync / backup 四线程的直接 `join()`,改复用 detach 路径已有的 `_stop_storage_threads()`:先以队列哨兵唤醒等在队列上的线程,再按每线程 `timeout=10` 顺序 join(总等待可达约 40 秒),join 完成后若仍有线程存活则抛 `RuntimeError`;确认全部停止后才清空共享状态并重启存储线程。

```diff
     def reset(self):
-        self.storage_stop_event.set()
+        # Reuse detach's queue wakeups and bounded joins, and fail if any
+        # storage thread remains alive before clearing shared state.
+        self._stop_storage_threads()
 
         self.write_queue.clear()
         self.load_queue.clear()
```

## 影响与回归风险
- 回归面:reset() 的线程停止语义变化:从「无限等待」变为「最多约 40 秒后抛错」;生产中若真有线程停滞,reset 将失败并抛 RuntimeError 而非挂死调度器。哨兵只能唤醒等在队列上的线程,不能中断进行中的 `all_reduce()`;join 超时也不会终止 worker 或取消其集合通信。队列清空、stop-event 清除与线程重建等成功路径逻辑保持不变(正文明确)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
