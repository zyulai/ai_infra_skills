# PR #35255: Fix: abort handling for dispatched requests after client disconnect

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#35255](https://github.com/sgl-project/sglang/pull/35255) |
| 作者 | Dovis01 |
| 合并日期 | 2026-09-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 20 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
客户端断连后已派发请求因状态被立即删除而逃过 abort、持续解码浪费 GPU,改为标记 dispatched、abort 后保留状态至调度器清理完成。

## 问题现象
客户端断开后,请求仍可能在调度器中继续解码,浪费 GPU 资源,并反复输出日志:`Received output for rid=... but the state was deleted in TokenizerManager.`。

## 根因
PR 正文披露:请求 handler 在 dispatch 之后被取消时,TokenizerManager 会立即删除对应的 `rid_to_state` 条目;而断连 abort 路径在发送 AbortReq 之前会先检查该状态,状态已删则 abort 被跳过,请求在 scheduler 中继续运行。另有一个相关竞态:chunked-prefill 请求若在其延迟 abort 被处理前已移出 chunked-prefill 槽位,abort 会丢失。

## 修复方案
TokenizerManager(`tokenizer_manager.py`):
- `ReqState` 新增 `dispatched` 与 `abort_sent` 标志;`_send_one_request`/`_send_batch_request` 派发成功后调用 `_mark_state_dispatched` 记录;
- `generate_request` 失败路径改用 `_release_req_states_on_failure(request_rids)`:未派发的 state 本地删除,已派发的调用 `abort_request` 并保留 state 直到调度器响应路径清理;并行采样的生成 rid 一并纳入跟踪(`request_rids`);
- `abort_request`:用 `abort_sent` 去重重复 abort 与 abort 指标;dispatch 抛异常时回滚 `abort_sent` 以保留重试行为;
- `create_abort_task` 的断连回调按 rid 列表逐个检查 `rid_to_state` 后再 abort。

Scheduler(`scheduler.py`):
- `process_pending_chunked_abort`:请求已 finished 或不再持有 KV 时清除标记并返回;若请求已移到其他调度器队列,则清除延迟标记并对其当前位置重新调用 `abort_request`(修复 abort 丢失竞态)。

测试:新增 `test_scheduler_chunked_abort_race.py`,重写/扩充 `test_tokenizer_manager_rid_cleanup.py`(断连清理、abort 去重、dispatch 失败、并行采样、chunked-prefill 迁移);`test_chain_read_ratchet.py` 的小工具扩展 `for` 循环字面量键枚举以适配新代码(由 diff 推断)。

关键 diff 摘录:

```python
def _release_req_states_on_failure(self, rids: Iterable[str]):
    for rid in rids:
        state = self.rid_to_state.get(rid)
        if state is None:
            continue
        if state.dispatched:
            try:
                self.abort_request(rid)
            except Exception:
                logger.exception("Failed to abort request %s during cleanup", rid)
        else:
            del self.rid_to_state[rid]
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:TokenizerManager 的请求生命周期管理(初始化、失败清理、abort、断连回调)与 scheduler 的 chunked abort 处理。行为变化:已派发请求在 handler 失败时不再静默丢弃状态而是发 abort;`abort_request` 现在对同 rid 幂等(`abort_sent` 去重,由 diff 推断);断连回调对批量/并行请求的每个 rid 都尝试 abort。原 `_discard_pending_req_states` 被 `_release_req_states_on_failure` 取代,直接调用旧函数的下游代码会失效(由 diff 推断,无此类调用见 diff)。

## 关联
正文未列出(仅给出原始 commit `f9ad78ba2471abbc9a55ff4b4b1f3fa88a798a9e`)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
