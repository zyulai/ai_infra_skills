# PR #46091: [Bugfix] [Parser] Fix empty tool block silently dropping subsequent content

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46091](https://github.com/vllm-project/vllm/pull/46091) |
| 作者 | bbrowning |
| 合并日期 | 2026-06-18 |
| labels | bug, ready, tool-calling |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
空 tool block 使 parser 卡在 TOOL_PREAMBLE 并丢弃后续输出;补状态转移并修引擎 deferred content 释放。

## 问题现象
空 tool block(只有开/闭 tool 标记、中间没有解析出的函数或只有无意义内容)导致其后所有内容被静默丢弃。作者先用 Codex CLI(Qwen 3.6-27B)评审 vLLM 代码时发现该理论 bug;随后 Claude Code(Nemotron 3 Super)评审 GLM 4.7 parser 时,模型输出 `<tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`,恰好发出正确的 `<tool_call>`/`</tool_call>` token id 但中间无有意义内容,之后卡住不再输出,无意间踩中此 bug。因为流式场景无法缓冲完整 tool call 再判断,parser 仍按 tool call 解析,但空 tool call 之后的内容此前全部被丢弃。

## 根因
正文披露:Qwen3 与 Gemma4 parser 缺少 `TOOL_PREAMBLE → TOOL_END` 状态转移,空 tool block(开标记紧跟闭标记)无法恢复到 `CONTENT` 状态,停留在 `TOOL_PREAMBLE` 并静默丢弃全部后续输出;MiniMax M2 已有该转移。另外引擎侧 deferred content 的释放条件不完整:当 tool event 没有产生实际 delta 时,被扣下的内容不会被释放发出。

## 修复方案
- `vllm/parser/qwen3.py`、`vllm/parser/gemma4.py`:新增 `(ParserState.TOOL_PREAMBLE, "TOOL_END") → ParserState.CONTENT` 转移,由 `EventType.TOOL_CALL_END` 触发。
- `vllm/parser/engine/parser_engine.py` `_events_to_delta`:释放条件由 `if self._deferred_content and not seen_tool_event` 改为
```python
if self._deferred_content and (not seen_tool_event or not tool_call_deltas):
    content_parts.insert(0, self._deferred_content)
    self._deferred_content = ""
```
即 tool event 未产生实际 delta 时也把 deferred content 放回输出。
- `tests/parser/engine/trace_builder.py`:新增 `empty-tool-block` 场景(空 tool block 后跟 content),覆盖所有使用新引擎的 parser 与多种 token-per-delta 大小;MiniMax M2 段生成器补同样的开/闭标记注入。

单元测试 `2333 passed, 2 warnings`(tests/parser/engine/ 及 qwen3/gemma4/minimax_m2 等相关 tool/reasoning parser 测试)。

## 影响与回归风险
- 回归面:Qwen3/Gemma4 parser 状态机转移表、引擎 `_events_to_delta` 的 deferred content 释放逻辑;新增场景对所有走 trace_builder 的 parser 生效;MiniMax M2 仅新增测试。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
