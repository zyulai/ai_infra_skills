# PR #47546: [Fix] tokenizer_manager: Do not dispatch a request whose abort was already sent

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47546](https://github.com/sgl-project/sglang/pull/47546) |
| 作者 | metamergebot |
| 合并日期 | 2026-10-31 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论
请求在 tokenizer manager 预处理中被 abort 后仍会被派发并跑完;派发前检查 abort_sent,跳过并按 abort 收尾。

## 问题现象
若请求在 tokenizer manager 仍在准备阶段(例如抓取图片或发布多模态特征)时被 abort,`AbortReq` 会先于请求本身到达 scheduler:scheduler 丢弃它尚未收到的 rid 的 abort 且不留记录,而 `abort_request` 因 `state.abort_sent` 已置位也不会重发。请求随后照常派发并跑到结束,调用方却一直在等。普通部署下用一个慢图片 URL 加客户端断连即可复现:请求会跑满 `max_new_tokens`。PD 分离部署下,处于此状态的 decode 请求会一直占住 KV 直到 transfer timeout。正文给出的时序:1) rid 注册、预处理进行中;2) `abort_request` 置 `abort_sent` 并发送 `AbortReq`,scheduler 尚无该 rid 丢弃之;3) 预处理完成,`_send_one_request` 派发请求;4) 后续所有 `abort_request` 调用(含断连 watcher)在 `if state.abort_sent` 处提前返回。

## 根因
正文披露:abort 与派发之间存在竞态窗口——abort 在最后一个 `await` 期间落地时,`AbortReq` 比 TokenizedObject 早到 scheduler 被丢弃,而 tokenizer manager 侧 `abort_sent` 已置位导致既不重发、派发前也不检查,「已 abort」的请求最终仍被派发且无法再被中止。

## 修复方案
改 `python/sglang/srt/managers/tokenizer_manager.py`:
1. `_send_one_request` 与 `_send_batch_request` 在最后一个 `await` 之后检查 `abort_sent`,命中则跳过派发(检查之后全部同步执行,不会再插入 abort);跳过请求的已备运输特征按失败派发同样方式释放——CUDA VMM 经 `cancel_for_dispatch`,CUDA IPC feature proxy(`feature`/`precomputed_embeddings`)经池的 `cancel_proxy`;批量路径抽出为 `_drop_aborted_before_dispatch`,批内其余请求照常派发:
```diff
+            if self._skip_dispatch_if_aborted(tokenized_obj.rid):
+                self._cancel_cuda_ipc_features((tokenized_obj,))
+                return
```
2. `_wait_one_response` 对被跳过的请求经 `_handle_abort_req` 收尾,产生与 scheduler 弹出排队中 aborted 请求相同的输出(finish reason `abort`、「Abort in waiting queue」)并从 `rid_to_state` 移除 rid。
3. `abort_request(abort_all=True)` 把所有未派发请求标记 `abort_sent`,但新增 `ReqState.waiting_at_pause_gate` 字段豁免停在 `generate_request` 暂停门上的请求(暂停解除后照常运行)。
正文明确未覆盖:multi-tokenizer 模式(`tokenizer_worker_num > 1`),需 scheduler 侧记录才能覆盖。新增 `TestAbortBeforeDispatch` 三个单测,无修复时均以 waiter 超时失败。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:tokenizer manager 所有派发路径(单请求/批量/abort_all/暂停门)与请求收尾语义;新增每请求一次字典查询的开销;`_skip_dispatch_if_aborted` 会 set `encoder_dispatch_ready` 事件;未覆盖的 multi-tokenizer 模式行为不变;`test_request_class_override_points.py` 的 stub 状态新增 `abort_sent` 字段。

## 关联
正文未列出(仅附 CI 运行链接:Run #41156997876、Run #41156997912、Run #41156997782)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
