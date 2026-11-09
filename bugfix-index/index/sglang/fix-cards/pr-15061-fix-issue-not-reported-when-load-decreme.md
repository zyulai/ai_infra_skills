# PR #15061: Fix issue not reported when load decrement is incorrect

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15061](https://github.com/sgl-project/sglang/pull/15061) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-12-14 |
| labels | run-ci, model-gateway |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

网关 load 计数减到负值时被静默吞掉;decrement_load 改为告警日志报告异常递减。

## 问题现象

正文未描述具体现象(Motivation、Modifications 等小节均为空模板)。从标题看:当 load 计数被错误地递减(计数已为 0 还继续减)时,这个问题不会被报告出来,处于静默状态。

## 根因

正文未披露根因。由 diff 可描述机制:`sgl-model-gateway` 中 `BasicWorker.decrement_load` 通过 `load_counter.fetch_update(..., |current| current.checked_sub(1))` 做递减,当计数已为 0 时 `checked_sub` 返回 `None`,`fetch_update` 返回 `Err`;原代码用 `.ok()` 把这个错误静默吞掉,既不修正计数也不输出任何日志,异常递减因此无从发现(由 diff 推断)。

## 修复方案

改 `sgl-model-gateway/src/core/worker.rs` 的 `decrement_load`:把 `.ok()` 改为对 `Err` 分支显式告警,其余逻辑不变:

```diff
-        self.load_counter
+        if self
+            .load_counter
             .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                 current.checked_sub(1)
             })
-            .ok();
+            .is_err()
+        {
+            tracing::warn!(
+                worker_url = %self.metadata.url,
+                "Attempted to decrement load counter that is already at 0"
+            );
+        }
         self.update_running_requests_metrics();
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 Rust 网关 `BasicWorker.decrement_load` 一处;计数本身不做修正(仍是递减失败即无操作),只是新增一条带 worker_url 的 warn 日志,并照常更新 running_requests 指标;对正常递减路径无行为影响。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
