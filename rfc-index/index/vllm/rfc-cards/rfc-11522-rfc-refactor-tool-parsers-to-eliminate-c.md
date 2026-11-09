# RFC #11522: [RFC]: Refactor tool parsers to eliminate coding errors and allow more efficient implementations.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11522](https://github.com/vllm-project/vllm/issues/11522) |
| 作者 | elementary-particle |
| 创建日期 | 2024-12-26 |
| 状态 | ⚪ closed(关闭于 2025-09-05) |
| labels | RFC, stale, tool-calling |
| 热度 | 👍 2 · 💬 13 |
| 分类 | Architecture / Engine |

## 一句话结论
提议重构 tool parser:把输出状态维护职责移入 parser 自身,修复现有解析 bug,并支持基于增量状态的高效实现,消除长 tool call 的性能瓶颈。

## 提案动机
当前 tool parser 在使用中有 bug、代码混乱,尤其是 `extract_tool_calls_streaming` 的实现。许多已实现的 tool parser 在正确性与效率上都欠考究,给仓库带来大量 issue。从长期看,chat streaming server 里维护整个输出字符串、并为每个生成 token 反复解析全部输出的做法会变得非常昂贵,很快会在长 tool call 场景成为性能瓶颈。作者认为在加入更多 tool calling 支持之前是重构的最佳时机。

## 提出的方案
- 架构上明确:维护所需状态是 tool parser 自己的职责——如果 parser 需要完整输出,应在自己的类里用 `delta_text` 与 `delta_token_ids` 维护,而不是依赖 server 缓存。
- 这种职责划分鼓励 parser 使用自定义的高效替代实现来生成 delta 消息,例如基于内部 JSON parser 状态生成增量,而不必每次解析完整输出字符串。
- 无用户能力损失;用户界面无需改动,chat server 只需少量修改(仅移除输出缓存部分),主要重构发生在 tool parser 侧。
- 可渐进推进:先把字符串缓存移入 parser,再逐步重构。
- 任务清单:为现有 tool parser 增加内部输出缓存并修改 parser 接口;修复/重构现有 parser 的解析逻辑;为 tool parser 补充相应测试。

## 优化点与收益
正文未给出量化数字。收益为:消除「每个生成 token 都重新解析完整输出」的重复开销,长 tool call 场景的流式解析不再成为性能瓶颈;parser 可基于内部状态(如 JSON parser 状态)直接产出 delta 消息,实现更高效;同时修复正确性 bug。作者称「No capability is lost. Performance gains are good」。

## 现状与落地
已关闭(2025-09-05)。正文以任务清单形式列出三项工作(接口与缓存、修复/重构、测试),反馈期 1-2 周;关闭时各项是否完成、由哪些 PR 承接,需查关联 PR。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
