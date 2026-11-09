# PR #24070: [BugFix] Fix rid_to_state leak for aborted queued requests

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24070](https://github.com/sgl-project/sglang/pull/24070) |
| 作者 | guoyuhong |
| 合并日期 | 2026-05-20 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
排队中请求被 abort 时 `rid_to_state` 条目不清理,重提交报 Duplicate request ID;在 `_handle_abort_req` 补删除。

## 问题现象
abort 所有请求(如权重更新、pause/resume 期间)时,正在 scheduler 等待队列里排队的请求被 abort,`AbortReq` 经 `_handle_abort_req` 送回 `TokenizerManager`,但该 handler 从不删除 `rid_to_state[rid]`,留下陈旧条目。之后用同一 request ID 重新提交时抛 `ValueError: Duplicate request ID detected: <rid>`,请求失败。正文指出该 abort-all 模式在强化学习训练工作流中尤其常见——权重更新或 rollout 同步频繁触发全局请求清空与重提交,尚未开始解码的排队 rollout 请求最易中招,导致下一个训练 step 因 duplicate RID 报错而失败。

## 根因
正文披露:running 与 queued 两种请求走完全不同的 abort 路径,queued 路径缺失清理。running 请求 abort 走正常 forward→output 流水线(`to_finish = FINISH_ABORT()` → `stream_output()` 发 `BatchStrOutput` → detokenizer 转发 → `_handle_batch_output()` 中执行 `del self.rid_to_state[rid]`,有清理);queued 请求 abort 从不进入 forward,scheduler 直接从 `waiting_queue.pop(i)` 发回 `AbortReq` → `_handle_abort_req()` 只置 `state.finished = True`、`state.event.set()`,缺少 `del self.rid_to_state[rid]`。两条路径产生不同消息类型(`BatchStrOutput` vs `AbortReq`),分发到不同 handler,而只有 `_handle_batch_output` 带清理。

## 修复方案
`python/sglang/srt/managers/tokenizer_manager.py` 的 `_handle_abort_req`:在向 `state.out_list` 追加输出、set event 之前补上一行 `del self.rid_to_state[recv_obj.rid]`,与 `_handle_batch_output` 对已完成请求的处理保持一致:
```diff
             "output_ids": output_ids,
             "meta_info": meta_info,
         }
+        del self.rid_to_state[recv_obj.rid]
+
         state.out_list.append(out)
         state.event.set()
```
另新增复现单测 `test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py`(mock TokenizerManager),覆盖 `_handle_abort_req` 清理、`_handle_batch_output` 完成时清理、`_init_req_state` 拒绝重复 rid、完成/abort 后重提交等场景;无修复时重提交复现 `ValueError: Duplicate request ID detected: resubmit_after_abort_rid`,修复后全部通过。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:tokenizer manager 的 abort 收尾路径。删除条目发生在 `state.out_list.append` 与 `state.event.set()` 之前,`state` 对象本身已先取出(正文修改说明与 diff 显示按变量继续使用),waiter 仍能收到 abort 输出;若同一 rid 存在并发 abort/完成消息,二次 `del` 可能抛 KeyError(正文未讨论该场景)。新增单测注册为 CPU CI(`base-a-test-cpu`,估时 15s)。

## 关联
正文未列出(仅附 CI 运行链接:Run #26086923103)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
