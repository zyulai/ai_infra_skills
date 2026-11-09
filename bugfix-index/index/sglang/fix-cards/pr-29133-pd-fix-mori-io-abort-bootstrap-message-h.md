# PR #29133: [PD] Fix MORI-IO ABORT bootstrap message handling

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29133](https://github.com/sgl-project/sglang/pull/29133) |
| 作者 | maning00 |
| 合并日期 | 2026-08-28 |
| labels | amd, run-ci |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
Mori decode 侧 ABORT 通知被 MORI_GUARD 校验当作畸形 bootstrap 消息丢弃;把 ABORT 提前派发并将房间标记为 Failed。

## 问题现象
PD 分离的 Mori(MORI-IO)传输路径中,decode 侧发出的 `ABORT` 通知在 prefill/bootstrap 侧被 `MORI_GUARD` 帧校验误判为畸形 bootstrap 消息而丢弃(正文 Summary)。正文未给出具体报错日志。

## 根因
正文披露:decode 侧的 `ABORT` 通知因为要过 `MORI_GUARD` 帧校验,被当作畸形 bootstrap 消息处理,ABORT 得不到正确派发;同时 abort 之后还存在状态/失败记录(stale status / failure-record)清理问题。

## 修复方案
改动集中在 `python/sglang/srt/disaggregation/mori/conn.py`:

1. 新增 `_TAG_ABORT = b"ABORT"` 与 `_handle_abort_message`:解析 msg[1] 中的 bootstrap_room,在 `transfer_lock` 下检查 `request_status`——未跟踪或已是 `KVPoll.Success`/`Failed` 时忽略,否则 `update_status(room, KVPoll.Failed)`。
2. bootstrap 线程在 `recv_multipart` 后先判断首帧 tag,`ABORT` 消息在受 MORI_GUARD 保护的 payload 校验**之前**派发:

```diff
                     msg = self.server_socket.recv_multipart()
+                    if not msg:
+                        continue
+
+                    tag = msg[0]
+                    if tag == _TAG_ABORT:
+                        self._handle_abort_message(msg)
+                        continue
+
                     payload = self._validate_message(msg)
```

3. decode_worker 收状态时,房间已清理(`bootstrap_room not in self.request_status`)则丢弃迟到状态,避免 stale status。
4. `add_transfer_request`:房间不存在或已 Failed 时直接返回,不再发起新传输。
5. `abort()`:`super().abort()` 并 `clear()` 之后,在 `failure_lock` 下 `failure_records.pop(bootstrap_room, None)`,清理失败记录。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:仅影响 Mori PD 分离传输路径:ABORT 提前派发、房间状态置 Failed、迟到状态丢弃、abort 后失败记录清理;`_validate_message` 对带 MORI_GUARD 的正常 payload 校验逻辑不变。非 Mori 后端(mooncake 等)不受影响。若 decode 侧发送的 ABORT 帧格式与 msg[1] 为 ascii 房间号的约定不符,会记录 warning 并忽略。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
