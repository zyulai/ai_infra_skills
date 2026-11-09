# PR #49206: fix: resolve silent request skipping in PRIORITY scheduling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49206](https://github.com/vllm-project/vllm/pull/49206) |
| 作者 | Tejas-Raj01 |
| 合并日期 | 2026-08-06 |
| labels | ready, v1 |
| 热度 | 👍 3 · 💬 7 |
| 分类 | Overlap / Scheduler |
| bug 类型 | correctness |

## 一句话结论
PRIORITY 策略抢占路径在被打断请求位于遍历游标之前时未正确回正 req_index,导致后一请求整步被静默跳过;修复为按受害者位置精确调整游标。

## 问题现象
v1 调度器在 `SchedulingPolicy.PRIORITY` 策略下发生抢占时,后续请求会在整个调度步中被静默跳过(不被调度也不报错)。回归场景(#49097):A(优先级 9,最低)、B(优先级 0)、C(优先级 1)依次入队,在 KV 压力下触发抢占后,修复前的回归测试断言失败:`AssertionError: Bug present: Request C was silently skipped...`,且 C 的状态未回到 RUNNING。

## 根因
正文披露:请求被抢占时,如果被抢占请求在 `self.running` 列表中位于当前迭代游标(`req_index`)之前,调度器的请求簿记(`req_index`)没有被正确调整。旧代码在按优先级重排后用 `self.running.remove(preempted_req)` 移除请求,随后无条件地 `req_index -= 1`;但当被移除请求的下标大于等于当前游标时(即本次抢占选择的是排在后面的请求),这次递减是错误的,游标整体前移一格,导致紧接着的一个请求在遍历中被跳过。游标错位与无条件递减的组合从 diff 中可确认(由 diff 推断)。

## 修复方案
改 `vllm/v1/core/sched/scheduler.py` 的 `schedule()` PRIORITY 抢占分支:
```python
-                        self.running.remove(preempted_req)
+                        victim_index = self.running.index(preempted_req)
+                        del self.running[victim_index]
+                        if victim_index < req_index:
+                            req_index -= 1
```
先记录被抢占请求的确切下标,按位置删除;只有当受害者位于当前游标之前时才把 `req_index` 减一,否则不动游标。同时删除了后面分支里无条件的 `req_index -= 1`。新增回归测试 `tests/v1/core/test_priority_preemption_bug.py`(构造 KV 压力下 A/B/C 三请求的抢占场景)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(labels v1)。
- 回归面:仅 v1 调度器 `schedule()` 中 PRIORITY 策略的抢占簿记;非 priority 策略走 `self.running.pop()` 的 else 分支不受影响。作者跑过 `tests/v1/core/test_scheduler.py` 确认无回归。

## 关联
Fixes #49097

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
