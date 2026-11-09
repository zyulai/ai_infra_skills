# PR #46159: [Bugfix][Parser] Fix U+FFFD leak at reasoning-to-content transition in engine parsers

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46159](https://github.com/vllm-project/vllm/pull/46159) |
| 作者 | bbrowning |
| 合并日期 | 2026-06-19 |
| labels | bug, ready, tool-calling |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
byte-fallback token 跨 delta 边界时逐 token decode 产生 U+FFFD 泄入流式 content;改为在 reasoning→content 切换处 flush 引擎 lexer 修复。

## 问题现象
流式输出中,Unicode 替换字符(U+FFFD)在 reasoning→content 转换处被泄漏给客户端。手动复现:服务 GLM 4.7 系模型(`zai-org/GLM-4.7-Flash`,配 glm47 tool parser 与 glm45 reasoning parser),循环发送可能触发多字节 Unicode 字符跨越转换边界的请求(如韩文提问「삼성전자 주주총회는 어디에서 열려?」);修复前约 20-40% 的请求会在转换处看到 U+FFFD 泄漏,修复后完全不再出现。该问题最初在 #45919 的评论中被识别。

## 根因
正文披露:在 reasoning→content 转换处,原先通过对孤立的 content 侧 token ID 逐个调用 `tokenizer.decode()` 得到文本;当 byte-fallback token 跨越 delta 边界时,孤立解码这些 token 会产生 U+FFFD。修复方式是改为在转换处 flush reasoning parser 的引擎 lexer,而不是解码孤立的 token ID。

## 修复方案
1. `vllm/parser/abstract_parser.py` 的 `parse_delta`:engine 模式下转换发生时,先调用 `reasoning_parser.finish_streaming()` 取得 flush_delta,content 文本改用 `delta_message.content` 与 `flush_delta.content` 拼接,不再 `decode(current_token_ids)`;
2. `vllm/parser/engine/parser_engine.py` 的 `finish_streaming`:当 `events` 或 `_deferred_content` 存在时也返回 delta(以 `finished=True` 转换事件),保证被推迟的 content 被冲出;
3. 新增回归测试 `tests/parser/engine/test_ufffd_reasoning_transition.py`(用模拟 byte-fallback tokenizer 在多种 chunk size 下断言无 U+FFFD 泄漏)。

关键 diff 摘录:
```python
flush_delta = reasoning_parser.finish_streaming()
current_text = (
    (delta_message.content if delta_message else None) or ""
) + ((flush_delta.content if flush_delta else None) or "")
```

## 影响与回归风险
- 回归面:engine 侧 parser 的 reasoning→content 转换路径(`DelegatingParser.parse_delta`、`ParserEngine.finish_streaming`),影响所有基于引擎 lexer 的 reasoning/tool parser(GLM4.7、Qwen3 等);非 engine 模式分支不变。

## 关联
- [PR #45919 评论](https://github.com/vllm-project/vllm/pull/45919#issuecomment-4732432638)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
