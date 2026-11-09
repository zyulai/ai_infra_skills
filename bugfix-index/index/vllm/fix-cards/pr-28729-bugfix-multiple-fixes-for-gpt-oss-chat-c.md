# PR #28729: [Bugfix] Multiple fixes for gpt-oss Chat Completion prompting

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28729](https://github.com/vllm-project/vllm/pull/28729) |
| 作者 | bbrowning |
| 合并日期 | 2025-12-12 |
| labels | frontend, ready, tool-calling, gpt-oss |
| 热度 | 👍 13 · 💬 31 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
gpt-oss Chat Completion 的 Harmony 消息转换有 7 处缺陷(工具响应 recipient/id 映射、reasoning/commentary 恢复、analysis 消息未清理);重写解析逻辑后 BFCL 多轮准确率 45.0%→55.0%。

## 问题现象
gpt-oss 模型在 Chat Completion 多轮(尤其带工具调用)场景下提示词构造错误,表现为:模型搞不清哪个 tool response 对应哪个 tool call;上一轮的 commentary 内容与 reasoning 没有被还原进正确通道;final 输出之前的 analysis 消息(原始思维链)常被带给模型,造成 token 用量增加、上下文污染,且不符合 OpenAI 对原始思维链处理的最佳实践。BFCL v4 multi_turn 基线(2025-11-12 main,commit 304419576ae9)总体准确率 45.0%。

## 根因
正文逐条披露(共 7 项):工具响应未设置 recipient 为 "assistant";tool_call_id 未映射回函数名;工具调用轮次中返回的 commentary 通道 content 与 reasoning 未在后续轮次正确还原为 commentary/analysis 消息;非工具的 `reasoning` extra_body 内容未还原为 analysis 消息;非工具、非 reasoning 输出生成 Harmony 消息时未把通道标记为 `final`;final 通道输出之前的 analysis 消息未在所有情况下被丢弃。

## 修复方案
- `vllm/entrypoints/openai/parser/harmony_utils.py`:新增 `parse_chat_inputs_to_harmony_messages`(先收集 tool_call_id→函数名映射再逐条转换)、`parse_chat_input_to_harmony_message`(tool_calls → commentary 消息、reasoning → analysis 消息、tool 响应 → `functions.<name>` 作者 + recipient "assistant")、`auto_drop_analysis_messages`(丢弃最后一条 assistant final 消息之前的所有 analysis 消息)、`flatten_chat_text_content`;重写 `parse_chat_output` 按通道收集 reasoning/final 并把空串归一为 None。
- `serving_chat.py`:`_make_request_with_harmony` 改用新解析器并调用 `maybe_serialize_tool_calls`;流式路径把 commentary 通道的 preamble 作为 content 下发。
- `openai_tool_parser.py`:content 优先取 final,否则回退 commentary(tool call 前言)。
- 新增 11 + 72 个单测;pack 的 diff 在 40000 字符处截断(文件粒度),截断部分未见到。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(基线为 2025-11-12 的 main)。
- 回归面:gpt-oss 的 Chat Completion 入口路径(harmony 消息解析、流式增量、openai 工具解析器的 content 选择);非 Harmony 模型不受影响。正文未列局限性。

## 关联
正文未列出关联 issue/PR(基线 commit 304419576ae9dc2ecaa28c4506d3870f7c68bd85)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
