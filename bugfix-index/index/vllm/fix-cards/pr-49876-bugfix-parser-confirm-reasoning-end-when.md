# PR #49876: [Bugfix][Parser] Confirm reasoning end when an Inkling content block opens

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49876](https://github.com/vllm-project/vllm/pull/49876) |
| 作者 | Vegetog |
| 合并日期 | 2026-08-08 |
| labels | bug, frontend, tool-calling, verified |
| 热度 | 👍 4 · 💬 25 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
Inkling 响应以可见内容开头且无 thinking block 时不发 REASONING_END,DelegatingParser 不进工具阶段、工具调用原样漏进正文;在所有可见内容块开启转换上补发 REASONING_END。

## 问题现象
流式场景(先文本后工具调用)下,当 Inkling 响应以可见内容开头、且没有 thinking block 时,工具块被当作 assistant 正文原样返回,工具调用不会被解析。正文示例:模型输出含 `<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>`,而返回的 content 是 `let me check<|content_invoke_tool_json|>{...}<|end_message|>`,没有解析出 tool call。触发条件:两个 Inkling parser 均启用、声明了 tools、`tool_choice="auto"`。这是 #49865 的「先文本后工具」流式半边;#51391 修了 reasoning pass 内的 marker ownership,但没修这个 phase gating,main 上仍复现。

## 根因
正文披露:Inkling 的 reasoning pass 只在显式的 reasoning-end 事件上离开 reasoning 阶段;以可见内容开头、没有 thinking block 的响应永远不会发出该事件,于是 `DelegatingParser` 从未进入工具阶段(tool pass 不运行),工具块自然留在 content 里。

## 修复方案
`vllm/parser/inkling.py` 的 `inkling_config()` 中 6 处一行改动:凡是打开一个渲染为可见内容的块的转换,都补发 `REASONING_END`——`(CONTENT, TEXT_START)`、`(CONTENT, TOOL_TEXT)`、`(CONTENT, TOOL_ERROR)`、`(MESSAGE_HEADER, TEXT_START)`、`(MESSAGE_HEADER, TOOL_TEXT)`、`(MESSAGE_HEADER, TOOL_ERROR)` 的 Transition 输出由 `()` 改为 `(EventType.REASONING_END,)`。这些 opener 不是 tool terminal,与 #51391 引入的 forwarded-tool-span 跟踪不交互;parser/engine 其余部分未动。

```diff
         (ParserState.CONTENT, "TEXT_START"): Transition(
             ParserState.CONTENT,
-            (),
+            (EventType.REASONING_END,),
         ),
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:仅 Inkling parser 配置的状态转换表;THINK_START 等进入 reasoning 的转换不变。测试:`tests/parser/engine/test_inkling.py` 新增 `test_visible_text_then_tool_streaming`,按 chunk size 1/3/64 × 三个 opener(TEXT_START/TOOL_TEXT/TOOL_ERROR)共 9 个参数化用例,main 上全 FAIL、本 PR 全 PASS;`tests/parser/engine` 3767 passed(较 main 的 3758 恰好多 9 个)。作者按 review 收窄范围:去掉 `passthrough_terminal_when_skipping_tools` 及其 engine/config 管线、去掉通用 `_extract_tool_calls` clean-content fallback;无工具纯文本的流式延迟问题拆到后续 PR,tool-first/no-thinking 场景留给 #50528。

## 关联
- Fixes the streaming text-then-tool half of #49865
- 相关 PR:#51391(reasoning pass marker ownership)、#50528(TOOL_START → REASONING_END 转换)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
