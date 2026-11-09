# PR #7708: Fix the issue of incorrect finish reason in final stream response chunk returned during tool call

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7708](https://github.com/sgl-project/sglang/pull/7708) |
| 作者 | xianzhiT |
| 合并日期 | 2025-07-23 |
| labels | ready-to-merge, function-calling |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
流式输出以 tool_calls 结束时,末尾 chunk 的 finish_reason 恒为 stop;工具调用流路径回传并改写 finish_reason 为 tool_calls。

## 问题现象
自 v0.4.8 起,为符合 OpenAI 协议,`OpenAIServingChat` 会在流式输出末尾额外追加一个 chunk 说明结束原因。当流式输出以 tool_calls 结束时,该末尾 chunk 的 `finish_reason` 应为 `tool_calls`,实际却总是 `stop`。影响:依赖该字段的开源软件会出现不可预测行为(正文举例 strand agent)。

## 根因
正文披露:生成末尾 finish_reason chunk 的代码没有做工具调用判断——`_process_tool_call_stream` 产生的工具调用 chunk 直接 yield,末尾 chunk 沿用了上游传入的 `finish_reason_type`(通常为 `stop`),没有在判定为工具调用结束时将其改写为 `tool_calls`,违反 OpenAI 协议对 tool_calls 结束场景的约定。

## 修复方案
`python/sglang/srt/entrypoints/openai/serving_chat.py`:
1. `_process_tool_call_stream` 的 yield 由单值改为二元组 `(chunk, finish_reason_type)`,把工具调用路径判定的 finish_reason 一并传出;并在末尾追加:`if finish_reason_type == "stop": yield None, "tool_calls"`——工具调用场景下若仍为 stop,则返回空 chunk 加 `tool_calls` 结束原因。
2. `_generate_chat_stream` 中消费该二元组:非空 chunk 才 yield,并用 `tool_call_finish_reason_type` 覆盖 `finish_reason_type`,使最终说明性 chunk 携带正确的结束原因。
3. `test/srt/openai_server/function_call/test_openai_function_calling.py`:流式用户新增断言——最后一个 chunk 的 `finish_reason` 必须为 `tool_calls`。

```diff
-            yield f"data: {chunk.model_dump_json()}\n\n"
+            yield f"data: {chunk.model_dump_json()}\n\n", finish_reason_type
...
+        if finish_reason_type == "stop":
+            yield None, "tool_calls"
```

## 影响与回归风险
- 影响窗口:v0.4.8 引入末尾 finish_reason chunk 之后至本修复之前。
- 回归面:OpenAI 兼容 chat streaming 接口中带 tools/tool_choice 的请求流:工具调用流的最后一个 chunk finish_reason 由 `stop` 变为 `tool_calls`(行为变化正是修复目的);`_process_tool_call_stream` 的返回契约变为二元组,内部调用方已同步适配;非工具调用(普通 content)流不受影响。

## 关联
正文未列出(仅附三张截图说明前后行为)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
