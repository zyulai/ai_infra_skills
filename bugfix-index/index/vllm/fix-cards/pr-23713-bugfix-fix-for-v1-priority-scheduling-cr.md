# PR #23713: [Bugfix] Fix for V1 priority scheduling crashes at preemption

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23713](https://github.com/vllm-project/vllm/pull/23713) |
| 作者 | Hanchenli |
| 合并日期 | 2025-08-28 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
V1 优先级调度在 KV 不足抢占低优先级请求时未同步 scheduled_running_reqs,导致崩溃;在抢占分支补一次移除修复。

## 问题现象
V1 优先级调度(priority scheduling)场景下,当 KV cache 空间不足、需要按优先级驱逐低优先级请求时,调度器发生崩溃(对应关联 issue #23347 报告的现象)。正文描述:请求因低优先级且 KV 空间不足而被驱逐(evicted)时,`scheduled_running_reqs` 没有被更新。正文未给出具体报错日志与复现命令,但 PR 附带了端到端的调度器单元测试来复现该场景。

## 根因
正文披露:当请求因低优先级且 KV 空间不足被驱逐时,`scheduled_running_reqs` 未随之更新,被抢占请求仍残留其中,最终引发崩溃。(由 diff 推断)修复点位于 `schedule()` 的优先级抢占分支,在把 `preempted_req` 从 `self.running` 移除之后,同步把它从 `scheduled_running_reqs` 中移除,说明崩溃源于该集合与 `running` 状态不一致、后续调度输出引用了已不在运行队列中的请求。

## 修复方案
改动两处:

1. `vllm/v1/core/sched/scheduler.py` 的 `schedule()`:优先级抢占分支新增对 `scheduled_running_reqs` 的条件移除:
```diff
                         self.running.remove(preempted_req)
+                        if preempted_req in scheduled_running_reqs:
+                            scheduled_running_reqs.remove(preempted_req)
                     else:
                         preempted_req = self.running.pop()
```
2. `tests/v1/core/test_scheduler.py`:新增 `test_priority_scheduling_preemption_when_out_of_kv`,用极小 KV 空间(num_blocks=5)构造"低优先级请求占满 KV → 高优先级请求到达 → 触发抢占"的场景,断言低优先级请求进入 waiting、只有高优先级请求继续运行;并给测试辅助函数 `create_requests_with_priority` 增加 `starting_idx` 参数,避免两个请求的 request_id/prompt 冲突。

## 影响与回归风险
- 影响窗口:正文未说明(发生在启用 V1 优先级调度并触发抢占的场景,见 #23347)。
- 回归面:仅改动 V1 调度器 `schedule()` 的优先级抢占分支与测试文件;非优先级路径的抢占(`self.running.pop()` 分支)不受影响。新增移除带 `in` 存在性判断,对集合中无该元素的路径无副作用;测试辅助函数新增参数带默认值,不影响既有测试。

## 关联
- #23347(修复的 issue)
- #23346(测试用例来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
