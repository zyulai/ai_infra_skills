# PR #19357: fix: Properly return abort error for streaming requests if the abort is triggered by scheduler

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19357](https://github.com/sgl-project/sglang/pull/19357) |
| 作者 | pyc96 |
| 合并日期 | 2026-03-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
scheduler 触发 abort 时流式请求未返回错误,HTTP 层直接断流、客户端收到空响应;流式生成器识别 abort 类型 finish_reason,改为 yield 携带状态码/消息的错误事件。

## 问题现象
以 gpt-oss-120b(`--tp=8 --max-queued-requests=1 --enable-priority-scheduling --priority-scheduling-preemption-threshold=1000 --max-running-requests=1` 等)启动服务并发送多个流式请求时,scheduler 侧 abort 的请求会导致:服务端日志出现 "Exception in ASGI application",`_generate_chat_stream` 在 `prompt_tokens[index] = content["meta_info"]["prompt_tokens"]` 处抛 `KeyError: 'prompt_tokens'`;客户端则收到空响应、无任何错误信息,连接被终止。正文未量化影响面。

## 根因
正文未单列根因分析段;按正文描述与所附 traceback:abort 由 scheduler 触发时,流式 chunk 的 `meta_info` 不含 `prompt_tokens` 等字段,原代码也只按普通 finish_reason 处理,未识别 abort 类型,于是在组装流式响应时直接 KeyError,异常沿 ASGI 栈抛出,流中断前客户端已收到空响应。(由 diff 推断)abort 的 finish_reason 携带 `status_code`/`message`,需要独立分支处理。

## 修复方案
- `serving_chat.py::_generate_chat_stream`:`finish_reason` 改为 `content["meta_info"].get("finish_reason", None)`;当 `finish_reason_type == "abort"` 时,用 `create_streaming_error_response(finish_reason.get("message", "Generation aborted."), code.name, code.value)` 生成错误事件 yield 给客户端并 `break`,非 abort 保持原逻辑。
- `serving_completions.py::_generate_completion_stream`:同样读取 `finish_reason` 用 `.get` 容错,abort 时 yield 错误事件并 break。
- 新增单测 `test_streaming_abort_yields_error`(chat 与 completions 各一):mock abort chunk,断言流中含 error 事件(消息与状态码 500),流以 `data: [DONE]` 结束。

关键 diff(serving_chat.py):

```python
                    # If the abort is from scheduler.
                    if finish_reason_type == "abort":
                        code = finish_reason.get(
                            "status_code", HTTPStatus.INTERNAL_SERVER_ERROR
                        )
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        break
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面:OpenAI 兼容 chat/completions 流式接口的 abort 处理:abort 现在以 SSE 错误事件(默认 500)+正常 [DONE] 结束,而不是抛异常断流;`finish_reason` 读取由 `[...]` 改 `.get(...)`,缺失时不再抛 KeyError;非 abort 的正常流式路径逻辑不变(按 diff 如实陈述)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
