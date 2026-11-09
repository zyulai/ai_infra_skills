# PR #28341: [Tokenizer] Fix abort racing server crash when large amount of aborts

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28341](https://github.com/sgl-project/sglang/pull/28341) |
| 作者 | hanming-lu |
| 合并日期 | 2026-06-16 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论
scheduler 回传的 AbortReq 与请求正常完成在 TokenizerManager.handle_loop 竞态,直接索引 `rid_to_state[rid]` 抛 KeyError 并被视为致命错误拖垮整个 server;改 dict.get() 缺席即早退修复。

## 问题现象
大量客户端断连(mass client disconnects)时,批量 abort 触发 server 崩溃。正文说明:scheduler 按 rid 前缀匹配或 `abort_all` 会把单个 abort 扇出成大量回传的 AbortReq,放大竞态;handle_loop 中抛出的 KeyError 被当作 fatal 处理(进程被 kill),一个良性竞态就能打挂整个 server。

## 根因
正文披露:scheduler 回传的 AbortReq 会与同一请求的正常完成在 TokenizerManager `handle_loop` 中竞态——一条 batch output 可能已经完成该请求并删除 `rid_to_state` 中对应表项,随后 abort echo 才被处理;此时直接下标访问 `rid_to_state[rid]` 抛 KeyError,异常传出 handle_loop 被当作致命错误,进程被杀。

## 修复方案
改 `python/sglang/srt/managers/tokenizer_manager.py` 的 `_handle_abort_req`:把 `self.rid_to_state[recv_obj.rid]` 换成 `dict.get()`,取不到时打一条 info 日志并提前返回——完成路径已经处理过请求,没有剩余可 abort 的东西:

```diff
-        state = self.rid_to_state[recv_obj.rid]
+        state = self.rid_to_state.get(recv_obj.rid)
+        if state is None:
+            logger.info(
+                "Abort request for rid=%s not found in rid_to_state; "
+                "likely already finished/removed.",
+                recv_obj.rid,
+            )
+            return
         state.finished = True
```

health check 类 abort 的既有早退逻辑保持不变。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 `_handle_abort_req` 一处:abort echo 到达时若请求已完成,行为由「抛 KeyError 杀进程」变为「记日志后忽略」;请求仍在途时 abort 语义完全不变。理论上若 rid 因其他原因异常丢失(而非正常完成),现在只会静默跳过并留下一条 info 日志,不再有崩溃信号。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
