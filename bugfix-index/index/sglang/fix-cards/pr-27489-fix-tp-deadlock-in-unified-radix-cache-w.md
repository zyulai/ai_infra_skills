# PR #27489: Fix TP deadlock in unified radix cache writing_check / loading_check

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27489](https://github.com/sgl-project/sglang/pull/27489) |
| 作者 | ispobock |
| 合并日期 | 2026-06-08 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 3 |
| 分类 | KV Cache / Memory |
| bug 类型 | hang |

## 一句话结论
开启 HiCache 长跑后 `writing_check` 中集合通信错配导致永久 TP 死锁;去掉空队列早退,保证所有 rank 都进入 all_reduce。

## 问题现象
开启 HiCache 后长时间运行,`UnifiedRadixCache.writing_check` 处出现永久性 TP 死锁。py-spy 指纹:
- TP0: `all_reduce → writing_check → check_hicache_events → get_new_batch_prefill_raw → event_loop_overlap`
- TP1-3: `broadcast → broadcast_pyobj → recv_requests → event_loop_overlap`(已进入下一迭代)

集合通信错配触发 300s watchdog 后进程被杀。

## 根因
PR 正文披露:TP0 进入了 `writing_check` 内的 all_reduce,而 TP1-3 在 `ongoing_write_through` 为空时走了早退路径,推进到下一轮事件循环的 `recv_requests` broadcast——不同 rank 执行的集合通信操作不一致。跨 rank 的 `ongoing_write_through` 会发生分歧(例如 `write_backup` 在部分 rank 上因 host-pool 或 aux-pool 分配失败而返回 0)。`loading_check` 存在同样的反模式(在 all_reduce 之前对空 `ongoing_load_back` 早退),随本 PR 一并修复。

## 修复方案
`python/sglang/srt/mem_cache/unified_radix_cache.py`:删除两个早退分支,保证每个 rank 都进入后面的 all_reduce。空本地队列对 MIN-reduce 贡献 0,`while finish_count > 0` 循环自然成为 no-op,开销小且集合安全:
```diff
-        if len(self.ongoing_write_through) == 0:
-            return
-
+        # Every rank must enter the all_reduce below; ongoing_write_through can
+        # diverge across ranks (e.g. write_backup returning 0 on a subset).
...
-        if cc is None or not self.ongoing_load_back:
+        if cc is None:
             return
+        # Every rank must enter the all_reduce below; ongoing_load_back can
+        # diverge across ranks.
```

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:仅 `unified_radix_cache.py` 的 `writing_check` / `loading_check` 两个轮询函数,影响开启 HiCache 的部署。
- 正文明确的 Tradeoff:移除早退并未解决跨 rank `ongoing_write_through` 分歧的根源;修复后进程不再死锁,但分歧条目可能不被 drain,表现为 host-pool 变慢 / lock-ref 泄漏而非硬挂。作者计划由后续 PR 将 `write_backup` 的入队决策改为集合一致。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
