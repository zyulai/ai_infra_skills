# PR #37051: Fix priority preemption regression test in scheduler

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37051](https://github.com/vllm-project/vllm/pull/37051) |
| 作者 | ezylopx5 |
| 合并日期 | 2026-04-01 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 16 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论
被 skip 的优先级抢占测试假设单步即抢占;改为多步模拟推进运行请求直至块压力触发,确定性断言抢占低优先级者。

## 问题现象
`tests/v1/core/test_scheduler.py` 中的 `test_priority_scheduling_preemption` 此前因依赖重复 request id 的错误行为而"碰巧通过";重复 req id 问题修复后该测试开始失败,遂被标记 `@pytest.mark.skip("needs investigation")`,优先级抢占路径处于无回归保护状态。正文未描述生产环境的用户可见故障,本 PR 是测试修复而非产品代码修复。

## 根因
正文披露:旧测试期望单个 schedule 步之后立即发生抢占;而实际行为是,抢占发生在 running 请求继续推进、申请到块边界的额外 KV block 之后。旧测试没有模拟 model 执行的迭代推进,内存压力从未在正确的时机形成,断言自然不成立——这也是它当初要靠重复 req id 才"通过"的原因。即测试的时序模型错了,不是调度器逻辑有误。

## 修复方案
仅改 `tests/v1/core/test_scheduler.py`,删除 skip 标记并整体重写该测试为确定性多步回归:block_size=16、num_blocks=6(1 个 null,5 个可用)构造精确的块数学——Phase 1 低优先级 lo1(priority 5)先运行并跨过 32-token 块边界;Phase 2 高优先级 hi1(priority 0)到达并被接纳,两者同时 running、0 空闲块;Phase 3 hi1 需要第 3 块时触发抢占,断言被抢占的是低优先级 lo1、hi1 仍在 running:

```python
+    lo1_req = scheduler.requests["lo1"]
+    assert lo1_req.status == RequestStatus.PREEMPTED, (
+        "Expected low-priority 'lo1' to be preempted"
+    )
+    assert any(req.request_id == "hi1" for req in scheduler.running), (
+        "High-priority 'hi1' should still be running"
+    )
```

同时删除了旧测试末尾对"是否抢占"的二义性 if/else 分支,改为确定性断言。验证跑过 `test_priority_scheduling_preemption`、`-k "priority_scheduling and not ec_connector"`、ec connector 抢占用例及整个 `test_scheduler.py`。

## 影响与回归风险
- 影响窗口:正文未说明;skip 存在期间该场景无测试覆盖。
- 回归面:纯测试文件改动,`vllm/` 产品代码零变更,无运行时回归风险。效果是把优先级抢占(抢占 running 中最低优先级请求,而非最高优先级者)的语义固定为可回归验证的断言;若未来调度器抢占时机或选择策略改变,该测试会如实失败而非静默跳过。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
