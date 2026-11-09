# PR #13201: Fix bug: Incorrect variable used in rem_total_token_offset calculatio…

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13201](https://github.com/sgl-project/sglang/pull/13201) |
| 作者 | liuhuijiayou |
| 合并日期 | 2025-11-24 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
优先级抢占时 `rem_total_token_offset` 误用传入请求而非被抢占请求的 token 数,内存记账错误;改用 `running_req` 修复。

## 问题现象
正文未描述具体现象,仅写「For more details, see issue #13111」。从新增测试的说明看,影响为优先级调度发生抢占(preemption)时内存记账错误:调度器认为可用内存比实际少,后续请求可能无法调度或出问题,且每次抢占误差持续累积。

## 根因
正文未披露根因,仅链接 issue #13111。新增测试 `test_priority_scheduling_preemption_token_offset_calculation` 的 docstring(diff 内容)给出了详细说明(由 diff 推断):`schedule_policy.py` 第 700 行抢占逻辑在释放被抢占请求的 token 时,误用了传入请求 `req` 而不是被抢占请求 `running_req`:
```python
self.rem_total_token_offset -= self._get_running_request_total_token_offset(req)
# 应为 running_req
```
测试场景中:被抢占请求需释放 8000 tokens,而触发抢占的高优先级请求仅 1000 tokens,于是系统以为只释放了 1000 而非 8000 tokens,差 7000 tokens 记账错误。

## 修复方案
`python/sglang/srt/managers/schedule_policy.py` 的 `preempt_to_schedule` 中,遍历 `running_batch.reqs` 释放可抢占请求时把参数从 `req` 改为 `running_req`:
```diff
-                    self._get_running_request_total_token_offset(req)
+                    self._get_running_request_total_token_offset(running_req)
```
并在 `test/srt/test_priority_scheduling.py` 新增 E2E 测试:四个不同优先级/token 数的并发请求,断言全部 200 且执行顺序为 priority 100 → 50 → 1 → 0(被抢占者最后完成),只有 token 记账正确时第四个请求才可被调度。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `preempt_to_schedule` 一行变量修正,影响启用优先级调度且发生抢占时的 `rem_total_token_offset` 记账路径;修复后调度器对可释放容量的估计更准确,可能改变抢占后的调度结果(原先被错误压制的请求现在可正常调度)。非优先级调度路径不受影响。

## 关联
- [issue #13111](https://github.com/sgl-project/sglang/issues/13111)(正文指出的详细背景)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
