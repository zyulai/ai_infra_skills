# PR #46586: [Fix] srt: Promptly abort running requests on client disconnect (#46517)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46586](https://github.com/sgl-project/sglang/pull/46586) |
| 作者 | divakantpandey |
| 合并日期 | 2026-11-09 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 23 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
客户端断连后流式请求仍继续解码;通过在 finally 中关闭 generator 并对流式请求逐块检查断连,及时 abort 请求。

## 问题现象
`/v1/chat/completions` 流失去客户端后可能继续解码。uvicorn 下 main 已处理常见情况(断连取消 response task → cancellation 传到 `generate_request` → 其 cleanup abort 请求),但仍有两条路径泄漏:
1. send 失败时 Starlette 让 body iterator 悬挂,且 ASGI spec >= 2.4 时还会跳过 background abort task;没有人关闭 `generate_request`,请求一直跑到 `max_tokens`。
2. `_stream_one_response` 只对非流式请求检查 `is_disconnected()`,流式请求继续向死 socket 写 chunk。

GPU 验证(RunPod 1x A40 48GB、镜像 nightly-dev-20261031-a2cafa56、Qwen/Qwen3-0.6B、`max_tokens=16000` + `ignore_eos`,4 秒后 RST 断开 TCP):stream 与 non-stream 两种模式 main 与本 PR 均在 1.0s / 4.2s 后归零;正文说明 uvicorn(ASGI 2.3)下 main 本就停住这些请求,该 run 只证明不回归;本 PR 修复的 send-failure 路径需要 ASGI >= 2.4,由单测覆盖。

## 根因
正文披露两条泄漏路径的机制:Starlette 在 send 失败时让 body iterator 保持悬挂,ASGI spec >= 2.4 下还跳过 background abort task,导致无人关闭 `generate_request`;以及 `_stream_one_response` 的断连检查此前只覆盖非流式分支。早期修订版曾在请求状态消失后发送强制 `AbortReq`,但 scheduler 按前缀匹配 abort rid,可能误停无关请求(`job` vs `job-1`),该方案连同 `force` 标志被移除。

## 修复方案
- `serving_chat.py`:`_generate_chat_stream` 在 `finally` 中 `aclose` 内层 `generate_request` generator,`prepend_first_chunk` 在 `finally` 中关闭外层 generator,并新增 `StreamingResponse` 子类在流结束时关闭 body iterator。关闭触发 `generate_request` 现有 cleanup(`_release_req_states_on_failure`),且仅当 rid 仍映射到该请求状态时才 abort。
- `tokenizer_manager.py`:`_stream_one_response` 的逐输出断连检查扩展到流式请求,断连时 `abort_request` 并抛 `ValueError` 杀掉整个调用栈与 asyncio task。
- 新测试 `test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py` 的 `TestChatStreamDisconnect` / `TestChatDisconnectAbort`:spec 2.4 下 send 失败会 abort 并在 scheduler ack 后释放状态;spec 2.3 下 `http.disconnect` abort;关闭已完成的 `job` 流不发 abort、`job-1` 继续运行;无状态的 orphan 未完成输出不发 abort;断连流在下一 chunk 发出前 abort。send-failure 与 per-chunk 测试在 main 上失败。

关键改动摘录(`serving_chat.py`):

```diff
+class _ClosingStreamingResponse(StreamingResponse):
+    async def stream_response(self, send) -> None:
+        try:
+            await super().stream_response(send)
+        finally:
+            await self.body_iterator.aclose()
```

## 影响与回归风险
- 影响窗口:正文未给出明确版本/提交范围;send-failure 路径需 ASGI spec >= 2.4,uvicorn(ASGI 2.3)下 main 已处理常见情况。
- 回归面:改动覆盖 OpenAI chat 流式响应关闭路径与 `_stream_one_response`(断连检查现同样作用于流式分支);abort 仅在该 rid 仍属于本请求时触发,并有前缀相关请求(`job`/`job-1`)不被误杀的回归测试。另含顺带改动:`few_shot_gsm8k.py` 新增 `max_request_errors` 容错(错误请求记为 invalid,超过阈值才失败),AMD disaggregation PP 测试设置 `MAX_REQUEST_ERRORS = 5`(注释称 Mooncake 同节点 RDMA aux send 偶发 stall,待 issue 45849 修复后移除)。

## 关联
- [#46517](https://github.com/sgl-project/sglang/issues/46517)(本 PR 修复的 issue)
- [sgl-project/sglang#45849](https://github.com/sgl-project/sglang/issues/45849)(测试注释中提到的 Mooncake RDMA stall issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
