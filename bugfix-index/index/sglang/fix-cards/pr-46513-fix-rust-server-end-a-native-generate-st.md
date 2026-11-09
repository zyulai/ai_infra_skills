# PR #46513: [Fix] rust-server: End a native /generate stream on a scheduler abort frame, as Python does

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46513](https://github.com/sgl-project/sglang/pull/46513) |
| 作者 | metamergebot |
| 合并日期 | 2026-10-28 |
| labels | run-ci, bypass-fail-fast |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
Rust 前端把 scheduler abort 帧变成裸 error item 丢帧;改为像 Python 一样以 abort 终止帧结束流并按客户端 rid 命名。

## 问题现象
当 scheduler 以错误状态 abort 一个请求(disaggregated decode 的 KV transfer 失败、`logprob_start_len` 超出 prompt 长度等)时,Python 服务与 Rust 前端对 native `/generate` 流式请求的应答不一致。Rust 前端把同一 scheduler 帧变成流上的一个裸 `{"error": {"message", "code"}}` item 并丢弃该帧:按 code 对 in-band error item 分类客户端、router 或 proxy,会把 Rust 模式下 429/503 的 abort 误读为"过载",而 Python 报告的是一个带 abort 原因的正常完成请求。此外 abort 消息以本服务交给 scheduler 的带后缀 rid(`Rid::from_client`)命名请求,而非客户端发送的 rid,向客户端泄露了它从未见过的请求 id 后缀。

## 根因
正文披露:Rust 前端把 scheduler 的 abort 帧直接转换为错误 item 并丢帧,与 Python `TokenizerManager._handle_abort_finish_reason` 的行为不一致——Python 把 abort 作为正常终止帧输出(含已有输出、请求计数、`e2e_latency`、`meta_info.finish_reason` 的 abort 信息,后跟 `[DONE]`),仅非流式请求才转为 HTTP 状态;同时 abort 消息使用的是带服务端后缀的内部 rid。

## 修复方案
使 native Rust 流像 Python 一样在 abort 帧上结束,并以客户端 rid 命名 abort 消息,共 5 个文件:
- `core/error.rs`:`CoreError::RuntimeRejected` 新增 boxed 的 `output: Box<CoreOutput>`(含 finish reason 与计数);`from_runtime_rejection` 接收它,新增 `into_rejected_output` 供唯一需要把 abort 报告为输出的表面(native `/generate` 流)取回;
- `core/handle.rs`:`generation_event` 把 scheduler 终止输出传入 rejection;`RuntimeCall::recv/try_recv` 经新增 `name_client_rid` 从 abort 消息中剥离本服务的 rid 后缀(兼容 `{rid}` 原样与 `{rid=}`/`{rid!r}` repr 两种形式);
- `http/native_api.rs`:`generation_event_stream` 把每个事件经新 `stream_event` 映射,将携带 scheduler rejection 的 `Failed` 还原为 `Finished(output)`,流以正常终止帧(`meta_info.finish_reason` 为 abort、`e2e_latency` 已设置)与 `[DONE]` 结束;
- `message/finish_reason.rs`:新增 `replace_in_abort_message`,仅改写 abort 消息子串;
- `grpc/native_api.rs`:错误映射测试夹具补上新字段。

非流式响应不变;替换旧测试为 `scheduler_abort_ends_a_stream_on_a_frame_and_fails_unary`,新增 `status_abort_names_the_client_rid`。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:native `/generate` 流式表面的 abort 行为改变(从 error item 改为终止帧),依赖旧 error item 形状的客户端会受影响(有测试特意替换固化该旧形状的用例);unary、batch unary item、OpenAI surfaces 与 gRPC 仍按 abort 的 status/message 应答,与 Python 一致;每次终止 item 增加一次子串检查开销,错误对象多携带一个 boxed output。

## 关联
正文未列出关联 issue/PR。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
