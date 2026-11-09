# PR #26822: [bugfix] Fix online serving crash when text type response_format is received

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26822](https://github.com/vllm-project/vllm/pull/26822) |
| 作者 | cjackal |
| 合并日期 | 2026-01-16 |
| labels | bug, frontend, ready, tool-calling |
| 热度 | 👍 6 · 💬 10 |
| 分类 | Frontend / Serving |
| bug 类型 | crash |

## 一句话结论

`response_format: {"type": "text"}` 会生成零约束的 StructuredOutputsParams,产生不可调度参数致在线服务崩溃;改为仅在有真实约束时构造并加输入校验。

## 问题现象

在线服务收到 `"response_format": {"type": "text"}` 的 chat completion 请求时崩溃(对应 issue #26639)。修复后该请求返回正常文本响应;同时验证 `{"type": "json_object"}` 无回归,返回 JSON 对象响应。测试用例为 `openai/gpt-oss-120b`、max_tokens 2048 的 curl 请求。

## 根因

正文披露:此前代码会生成「unschedulable structured outputs params」——即对不可调度的 structured outputs 参数未做输入校验,`StructuredOutputsParams` 允许在零约束下构造。(由 diff 推断)具体路径:`to_sampling_params()` 只要 `response_format` 非 None 就先创建空的 `StructuredOutputsParams()`,对 `type: "text"` 没有任何分支填充约束,于是空参数对象被一路下发,最终在线服务崩溃;工具调用路径 `adjust_request` 同样先建空对象再赋值。

## 修复方案

- `vllm/sampling_params.py`:`StructuredOutputsParams.__post_init__` 新增校验,约束计数 `count < 1` 时抛 `ValueError("You must use one kind of structured outputs constraint but none are specified")`。
- `chat_completion/protocol.py` 与 `completion/protocol.py` 的 `to_sampling_params()`:改为先收集 `structured_outputs_kwargs` dict,仅当非空时才构造——无已有约束时新建 `StructuredOutputsParams(**kwargs)`,已有则用 `dataclasses.replace` 派生新实例(不再原地改字段);`type: "text"` 因此不再创建任何结构化输出参数。
- `responses/serving.py` 的 `create_responses()`:structural tag 预处理改为先 `isinstance(struct_out, StructuredOutputsParams)` 判定,再以 `replace` 生成新对象。
- `tool_parsers/abstract_tool_parser.py` 的 `adjust_request()`:直接 `StructuredOutputsParams(json=json_schema_from_tool)` 一步构造,不再建空对象后赋值。
- 新增回归测试 `test_response_format_text`。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:OpenAI 兼容层所有涉及 structured outputs 的请求路径(chat/completions、completions、responses、tool calling)。零约束的 `StructuredOutputsParams` 现在会直接抛 `ValueError`,任何此前依赖空参数对象作为占位的内部调用方都会转为报错;约束字段改为不可变的 `replace` 派生,若有代码依赖原地修改 semantics 需复查;`response_format` 与显式 `structured_outputs` 同时给出时改为 replace 合并,字段优先级行为与旧的原地覆盖可能有细微差异。

## 关联

- #26639(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
