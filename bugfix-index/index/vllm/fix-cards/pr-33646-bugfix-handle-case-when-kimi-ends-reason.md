# PR #33646: [Bugfix] Handle case when kimi ends reasoning with a tool call

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33646](https://github.com/vllm-project/vllm/pull/33646) |
| 作者 | koush |
| 合并日期 | 2026-02-27 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
Kimi 推理解析器复用 DeepSeek 实现,推理中以 tool call 结束时不输出 `</think>` 导致工具调用被当作推理文本、无法执行;为其新写专用解析器处理隐式结束。

## 问题现象
Kimi 可能不输出 think 结束 token 而直接开始 tool call。当 Kimi 在推理(reasoning)内部以 tool call 结束时,该输出仅被登记为 end-of-message:既没有解析出 tool call,工具调用也不会被执行;在流式与非流式下,工具调用内容都被当作 reasoning 返回。作者在 Kimi 2.5 上经 Opencode、Open WebUI 以及直接调用 chat completions API 测试复现与验证。

## 根因
PR 正文披露:Kimi 的 reasoning parser 目前直接基于 DeepSeek 的实现(`kimi_k2` 映射到 `DeepSeekV3ReasoningWithThinkingParser`)。但 Kimi 可以在不输出 think 结束 token 的情况下开始 tool call——即通过 `<|tool_calls_section_begin|>` 隐式结束推理。DeepSeek 解析器只认 `</think>` 作为推理结束标记,因此 tool call 本身被视为推理的一部分,不会被解析执行。

## 修复方案
不再复用 DeepSeek 解析器,为 Kimi 从头新写 `vllm/reasoning/kimi_k2_reasoning_parser.py` 的 `KimiK2ReasoningParser`(作者称继承 DeepSeek 的改法并不比重写更省事,重写可复用基础解析器代码),并在 `vllm/reasoning/__init__.py` 中将 `kimi_k2` 的注册项改为指向新解析器:

```diff
     "kimi_k2": (
-        "deepseek_v3_reasoning_parser",
-        "DeepSeekV3ReasoningWithThinkingParser",
+        "kimi_k2_reasoning_parser",
+        "KimiK2ReasoningParser",
     ),
```

新解析器的核心处理:`is_reasoning_end` / `is_reasoning_end_streaming` 在遇到 `</think>` **或** `<|tool_calls_section_begin|>` 时都判定推理结束;`extract_content_ids` 与 `extract_reasoning(_streaming)` 在没有 `</think>` 时按 tool section 起始 token 切分,把其后内容作为 content 返回而非 reasoning。另外支持通过 `chat_template_kwargs` 的 `thinking` 开关关闭思考模式(此时回退到 IdentityReasoningParser)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:仅影响 `kimi_k2` reasoning parser 注册路径与新增文件;此前依赖 DeepSeek 解析器行为的 Kimi K2 思考模式输出切分逻辑会改变(thinking 关闭时走 identity 解析器)。其他模型的解析器注册不受影响。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
