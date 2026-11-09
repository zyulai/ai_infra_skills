# PR #48098: [Bugfix] Fix parallel_tool_calls=null crash in Responses API from_request()

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48098](https://github.com/vllm-project/vllm/pull/48098) |
| 作者 | mahadrehmann |
| 合并日期 | 2026-07-15 |
| labels | bug, frontend, ready, tool-calling, verified |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
Responses API 收到 "parallel_tool_calls": null 时响应构造抛 Pydantic ValidationError 返回 400;from_request() 中把 None 解析为文档默认值 True 修复。

## 问题现象
向 `/v1/responses` 发送 `{"parallel_tool_calls": null}` 时,请求失败:`from_request()` 把 None 传给类型为非可选 `bool` 的 `ResponsesResponse.parallel_tool_calls`,响应构造期间抛出 Pydantic `ValidationError`,被错误处理捕获后返回 400 及原始 Pydantic 报错("Input should be a valid boolean [type=bool_type, input_value=None...]"),而不是像请求侧字段那样解析为文档默认值。触发条件:请求体中显式携带 null——OpenAI Python SDK 默认序列化时会剔除 None 字段,故 bug 只在 curl 等不丢弃 null 的客户端上出现。

## 根因
正文披露:`from_request()`(responses/protocol.py)直接把 `request.parallel_tool_calls`(可能为 None)传给非可选 bool 字段;且 Responses API 与 ChatCompletion 路径不同,从不调用 `tool_calls_utils.py` 的 `maybe_filter_parallel_tool_calls`(在 vllm/entrypoints/openai/responses/ 下零命中),是独立代码路径,#44955 的修复未覆盖到它。

## 修复方案
改 `vllm/entrypoints/openai/responses/protocol.py` 的 `from_request()`,None 时取请求模型字段的文档默认值:

```diff
-            parallel_tool_calls=request.parallel_tool_calls,
+            parallel_tool_calls=request.parallel_tool_calls
+            if request.parallel_tool_calls is not None
+            else ResponsesRequest.model_fields["parallel_tool_calls"].default,
```

并在 `tests/tool_use/test_responses_request_validations.py` 增加回归测试(True/False/None→True 参数化)。实测:修复前同请求返回 400,修复后 200 且响应体含 `"parallel_tool_calls":true`(Qwen/Qwen2.5-0.5B-Instruct,T4 GPU)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 Responses API 响应构造中 `parallel_tool_calls` 的取值:显式 null 由"崩溃"变为解析为默认 True,显式 True/False 不变(由 diff 推断);ChatCompletion 路径无改动。

## 关联
- #48097
- #44948(ChatCompletion 同类 bug)
- #44955(仅修复 ChatCompletion 路径)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
