# PR #30673: Fix non-existent abort mode in Scheduler.pause_generation and inline retract_all

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30673](https://github.com/sgl-project/sglang/pull/30673) |
| 作者 | fzyzcjy |
| 合并日期 | 2026-07-15 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
pause_generation 接受并不存在/未处理的 abort 模式;收紧为 assert 仅允许 in_place/retract,并把 retract_all 内联。

## 问题现象
正文未描述具体现象(正文仅含 CI 状态)。标题指出 `Scheduler.pause_generation` 存在「不存在的 abort 模式」问题。从新增测试看:`PauseGenerationReqInput(mode="abort")` 与缺省构造(默认即 abort)都必须被 scheduler 端 assert 拒绝;retract 模式需正确 release 并把请求重新入队。

## 根因
正文未披露根因。(由 diff 推断)`pause_generation` 原先不校验 `recv_req.mode`,「abort」这一不存在的模式会被静默接受:不进入 retract 分支(原条件限定 `mode == "retract"`),却照常清空 last_batch/cur_batch_for_debug,行为未定义。同时 `ScheduleBatch.retract_all` 包装方法把「release 全部 reqs + 清空 self.reqs」封装在批对象上,模块级 `retract_all` 又以 `retracted_reqs = reqs` 别名返回,返回值语义混乱。本次收紧校验并把 retract 逻辑内联到 scheduler。

## 修复方案
- `python/sglang/srt/managers/schedule_batch.py`:模块级 `retract_all` 改为返回 `None`(删除 `retracted_reqs = reqs` 别名与 return);删除 `ScheduleBatch.retract_all` 包装方法。
- `python/sglang/srt/managers/scheduler.py`:`pause_generation` 开头新增 `assert recv_req.mode in ("in_place", "retract")`;retract 分支条件由 `recv_req.mode == "retract" and not self.running_batch.is_empty()` 简化为 `not self.running_batch.is_empty()`,内联调用模块级 `retract_all` 后显式 `self.running_batch.reqs = []`。
- 测试 `test_scheduler_pause_generation.py` 同步重写:abort/默认模式断言被拒;retract 用例改为 patch 模块级 `retract_all`;删除 `test_abort_drains_overlap_queue`,新增空批不 requeue、overlap 队列 drain 等用例。

```diff
-        if recv_req.mode == "retract" and not self.running_batch.is_empty():
+        if not self.running_batch.is_empty():
             self.running_batch.filter_batch()
-                retracted_reqs = self.running_batch.retract_all(
-                    self.server_args, offload_kv=False
+                retracted_reqs = self.running_batch.reqs
+                retract_all(
+                    reqs=retracted_reqs,
+                    server_args=self.server_args,
+                    req_to_token_pool=self.running_batch.req_to_token_pool,
+                    token_to_kv_pool_allocator=self.running_batch.token_to_kv_pool_allocator,
+                    tree_cache=self.running_batch.tree_cache,
+                    hisparse_coordinator=self.running_batch.hisparse_coordinator,
+                    offload_kv=False,
                 )
+                self.running_batch.reqs = []
```

## 影响与回归风险
- 回归面:`pause_generation` 的合法 mode 收紧为 in_place/retract,其他值(含缺省 abort)现在直接 AssertionError,调用方需先适配;`ScheduleBatch.retract_all` 方法被删除、模块级 `retract_all` 返回值由 `List[Req]` 改为 `None`,属接口变化,依赖旧返回值/旧方法的调用方会受影响;retract 语义保持 offload_kv=False(PD decode rebootstrap 场景不变),并补充空批、overlap drain 等测试覆盖。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
