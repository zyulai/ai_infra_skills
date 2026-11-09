# PR #38909: [Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38909](https://github.com/vllm-project/vllm/pull/38909) |
| 作者 | yoke233 |
| 合并日期 | 2026-04-08 |
| labels | bug, ready, tool-calling |
| 热度 | 👍 5 · 💬 7 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
Gemma4 流式工具调用把缓冲 delta 拼回 current_text 致 HTML 标签重复,改为采用上游 current_text、缓冲仅用于发射。

## 问题现象
Gemma4 流式输出中,工具参数含 HTML 内容(如 `write_file(content=...)`)时,`<html>`、`<meta>` 等标签被流式输出成 `<<htmlhtml`、`<<metameta` 等畸形结果;工具调用结束后(或内部)的普通文本也会被破坏(如 `<div>` 变成内部 `<<div>`)。FIX #38910。

## 根因
正文披露:`Gemma4ToolParser.extract_tool_calls_streaming()` 为避免泄露不完整的特殊 token 而缓冲 `delta_text`,随后用 `current_text = previous_text + delta_text` 重建 current_text,把缓冲后的输出状态与原始累计模型文本混在一起。缓冲的 `<` 被回放进 current_text 后,解析器产生 `<<div>`、`<<html`、`<<meta` 等非法中间态,再与参数 diff 逻辑相互作用,最终流式工具参数里标签名被复制。

## 修复方案
`vllm/tool_parsers/gemma4_tool_parser.py` 中不再重建 current_text:

```diff
-        # Reconstruct current_text after buffering to stay in sync
-        current_text = previous_text + delta_text
+        # Keep current_text from the upstream stream state. The buffered delta
+        # is only for emission, and must not be stitched back into the
+        # accumulated model text or normal content like "<div>" can be
+        # duplicated into "<<div>" when a tool call just ended.
```

current_text 直接采用上游流状态,缓冲的 delta_text 仅用于发射。新增 `tests/tool_parsers/test_gemma4_tool_parser.py` 两条回归用例(工具调用后普通文本保持 `<div>`、HTML 参数不产生重复前缀);作者说明未在本地跑全量 pytest(环境未完全就绪)。

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:Gemma4 流式工具调用的 current_text 语义;后续判断(如 `tool_call_start_token not in current_text` 是否按内容发射)改基于原始累计文本,防泄露初衷仍由 delta_text 缓冲承担。

## 关联
[#38910](https://github.com/vllm-project/vllm/issues/38910)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
