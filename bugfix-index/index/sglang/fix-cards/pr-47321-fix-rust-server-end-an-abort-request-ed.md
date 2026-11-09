# PR #47321: [Fix] rust-server: End an /abort_request-ed request on the scheduler's abort finish reason, as Python does

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47321](https://github.com/sgl-project/sglang/pull/47321) |
| 作者 | metamergebot |
| 合并日期 | 2026-10-31 |
| labels | dependencies, run-ci |
| 热度 | 👍 0 · 💬 19 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
Rust server 的 /abort_request 复用 gRPC Abort 立即取消请求、不给部分输出;改为像 Python 一样等 scheduler 以 abort finish reason 的终止输出结束请求。

## 问题现象
Python 的 `POST /abort_request` 不会 fail 它所指的请求:`TokenizerManager.abort_request` 向 scheduler 转发 `AbortReq` 并保留请求的响应状态,scheduler 的下一步以终止输出结束该请求——包含已生成的部分文本、token 计数和 `finish_reason` `{"type": "abort", "message": "Aborted"}`。流式客户端看到 `finish_reason` 为 `abort` 的最终 chunk 然后 `[DONE]`;unary `/generate` 得到带部分文本的 200。而 Rust 前端的 `/abort_request`(#45888)复用了 gRPC `Abort` 路径,立即取消请求并以 cancelled 错误结束客户端调用:经 HTTP 路由 abort 的客户端拿到错误 body、没有部分输出,与 Python server 行为不一致。

## 根因
正文披露:Rust 的 `/abort_request` 直接复用了 gRPC `Abort` 路径,该路径立即取消请求并使客户端调用以 cancelled 错误结束,因此走 HTTP 路由的 abort 得不到 Python server 那种「以 abort 终止输出完成」的响应。

## 修复方案
按正文 Modifications:1) `handlers/generate.rs`:`CoreHandle::abort` 增加 `AbortEnding` 参数——`CancelledError` 保留 gRPC `Abort` 行为;`AbortFinishReason` 为 `/abort_request` 使用:尚未被 Intake 认领的请求在开始前取消并以 frontend abort output 结束;已接受的请求作为 `AbortSource::Request` 交给 scheduler,调用继续等待 scheduler 的终止输出;`Generation` 观察到的取消变为 `Cancellation`(`Error`/`Unstarted`/`Scheduler`)并记住一次观察。2) `handlers/handle.rs`:`RuntimeCall` 记录请求最近输出的 counts 与 cache split;scheduler 交办的 abort 得不到响应时(Intake 关闭 sink 或 scheduler 静默 30s)以重复这些计数的 frontend abort output 结束并标记 `abort_output`。3) `to_scheduler.rs`/`wiring.rs`:`AbortSource::Request` 保留 detokenizer entry,并把 `AbortReq` 排在请求之后入 ring(满时回退旧路径);`RequestAdmission` 增加 `cancel_if_pending`/`is_accepted`。4) `/abort_request` 用 `AbortFinishReason`,gRPC `Abort` RPC 保留 `CancelledError`。另:正文注明 14 行改动仅为适配 #46181 的 `handlers/` 目录拆分,无行为变化。cargo test 434 passed / 0 failed。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:Rust server 的 HTTP `/abort_request` 语义变化(客户端从 cancelled 错误变为带部分输出的 abort 完成响应);gRPC `Abort` 路径保持不变;request ring 的控制消息排队顺序。注:pack 的 diff 在 40000 字符处按文件粒度截断,可见部分为 Python 侧 SIGTERM drain/关闭流程与 gRPC `Abort` 调用点的改动,正文所述 `handlers/generate.rs` 等核心 Rust 改动不在可见 diff 范围内,本卡修复方案以正文为准。

## 关联
- https://github.com/sgl-project/sglang/pull/46961
- https://github.com/sgl-project/sglang/pull/45888
- https://github.com/sgl-project/sglang/pull/46181

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
