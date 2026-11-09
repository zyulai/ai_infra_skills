# PR #22223: Revert "[Bugfix] V1 Fix the cursor leakage issue during request scheduling."

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22223](https://github.com/vllm-project/vllm/pull/22223) |
| 作者 | WoosukKwon |
| 合并日期 | 2025-08-05 |
| labels | v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论
整体回退 #21173 的「调度期 cursor 泄漏修复」,恢复优先级抢占路径原先的 `running.remove` 处理并删除其回归测试。

## 问题现象
正文未描述具体现象,仅写明 "Reverts vllm-project/vllm#21173",即回退 V1 调度器「request 调度期间 cursor 泄漏」修复的全部改动。

## 根因
正文未披露根因(正文仅一句回退声明,未说明 #21173 引发了什么问题)。(由 diff 推断:回退把优先级抢占分支中基于 `preempted_index` 调整 `req_index`、并从 `scheduled_running_reqs` 移除被抢占请求的逻辑,恢复为简单的 `self.running.remove(preempted_req)`;同时删除了配套的回归测试 `test_priority_scheduling_preemption_victim_iterator_order` 及测试 helper 的 request_ids 参数,即 #21173 引入的内容全部撤销。)

## 修复方案
回退 #21173 的改动,涉及两处:
1. `vllm/v1/core/sched/scheduler.py` 的 `schedule()`:优先级抢占分支恢复为 `self.running.remove(preempted_req)`,删除对 `preempted_index`、`req_index` 调整与 `scheduled_running_reqs.remove(preempted_req)` 的处理;
2. `tests/v1/core/test_scheduler.py`:删除 `test_priority_scheduling_preemption_victim_iterator_order` 测试,`create_requests_with_priority` 恢复为不接受 `request_ids` 参数。

关键 diff 摘录(恢复后的核心逻辑):
```python
preempted_req = min(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
self.running.remove(preempted_req)
```

## 影响与回归风险
- 回归面:V1 调度器优先级抢占路径恢复到 #21173 之前的实现;#21173 原本针对的 cursor 泄漏问题在该路径上可能重新出现(正文未评估);被删除的回归测试不再守护该行为。

## 关联
- [PR #21173](https://github.com/vllm-project/vllm/pull/21173)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
