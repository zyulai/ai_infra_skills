# PR #45553: [Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45553](https://github.com/vllm-project/vllm/pull/45553) |
| 作者 | lucianommartins |
| 合并日期 | 2026-06-16 |
| labels | bug, documentation, ready, tool-calling |
| 热度 | 👍 3 · 💬 8 |
| 分类 | Attention / Kernel |
| bug 类型 | resource |

## 一句话结论
修 Gemma4 工具调用三处遗留 bug:离线解析参数截断、thinking 关闭时 delimiter token 泄漏、structured output 被绕过,并同步上游 chat template。

## 问题现象
三个已报告 bug:其一(#39069),离线 `_parse_tool_arguments` 对含内部双引号的字符串参数值发生截断;其二(#39130),`enable_thinking=false` 时 `is_reasoning_end()` 返回 `False`,静默绕过 xgrammar structured output;其三(#39588),`enable_thinking=false` 时 tool-call delimiter token 泄漏进 content(`tool_choice="none"` 时尤其明显)。另有 chat template 问题:参数为 `None` 时渲染错误、字符串类型 `function.arguments` 未拒绝、历史 assistant turn 缺 thought primer、`continues_into_next` 多余 `<turn|>`(#42776 报告)等。原 PR 初版含 11 个 bug,rebase 后 7 个已被 #45588(ParserEngine 重写)解决,本 PR 保留其余;#39130 的非空 fallthrough 情形仍未修、在本 PR 补全。

## 根因
正文披露:
- #39069:`_parse_tool_arguments` 先把 `<|"|>` 替换为 `"`,再用正则 `[^"]*` 提取值,遇到第一个内部引号即停止,造成截断。
- #39130:`input_ids` 中不存在 reasoning 标记时 `is_reasoning_end()` 走 fallthrough 返回 `False`(原为 `self._reasoning_ended`),thinking 关闭时结构性输出被跳过。
- #39588:`ParserEngine.adjust_request()` 无条件设置 `skip_special_tokens=False`;没有 reasoning channel token 需要保留时,delimiter token 就会进入 content。

## 修复方案
- `vllm/tool_parsers/gemma4_utils.py`:`_parse_tool_arguments` 删去自制的替换+正则+兜底逻辑,直接委托给 `vllm.parser.gemma4._parse_gemma4_args`(原生解析器可正确处理内部引号、嵌套对象、数组)。
- `vllm/parser/gemma4.py`:`Gemma4Parser.__init__` 从 `chat_template_kwargs` 读取 `_thinking_enabled`;覆写 `adjust_request`,thinking 关闭时不再设置 `skip_special_tokens=False`;`is_reasoning_end` 对 `<|turn>`/`<|tool_response>` 标记返回 `not self._thinking_enabled`,fallthrough 改为 `return True`(reasoning 从未开始即视为已结束)。
- `examples/tool_chat_template_gemma4.jinja`:与上游 HuggingFace 模板 PR 同步,含 `None`→`null`、字符串 arguments 用 `raise_exception` 拒绝、历史 turn 补 `<|channel>thought\n<channel|>` primer、`preserve_thinking` 默认 false、`add_generation_prompt` 守卫、`continues_into_next` 修复与 `.get()` 安全访问等。

```python
+    def adjust_request(self, request):
+        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
+        if not chat_template_kwargs.get("enable_thinking", True):
+            return request
+        return super().adjust_request(request)
```

## 影响与回归风险
- 影响窗口:正文未说明(限定在 #45588 合入后仍存在的 Gemma4 工具调用路径)。
- 回归面:Gemma4 工具调用/推理解析/chat template 渲染路径:离线参数解析结果与原生在线解析器一致化;thinking 关闭会话的 structured output 与 delimiter 处理改变(预期行为);模板默认 `preserve_thinking=false` 是用户可见变化。测试:121 passed(gemma4 tool parser / reasoning parser / chat template 三套),测试期望相应更新(`is_reasoning_end` 翻转为 True、模板测试 arguments 由字符串改 mapping)。

## 关联
- #39069
- #39130
- #39588
- #45588(ParserEngine 重写,部分修复来源;见其 [comment](https://github.com/vllm-project/vllm/pull/45588#issuecomment-4711324461))
- #42776

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
