# PR #42128: [Bugfix] Fix Gemma4ToolParser streaming float corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42128](https://github.com/vllm-project/vllm/pull/42128) |
| 作者 | abinggo |
| 合并日期 | 2026-05-14 |
| labels | bug, ready, tool-calling, verified |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
Gemma4ToolParser 流式输出时,以 `.` 结尾的裸数字被立即解析为 float(如 `108.`→`108.0`),后续真值 `108.2` 的流式 diff 只补发 `2`,客户端得到 `108.02`;partial 模式下扣留尾点值修复。

## 问题现象
`Gemma4ToolParser` 在流式(streaming)模式下产生错误的浮点值,典型例子 `108.2` 变成 `108.02`(对应 issue #42047)。触发条件:partial/streaming 模式下,一个以 `.` 结尾的裸数值(如 `"108."`)先被逗号终结,随后真正的十进制位到达。非流式路径不受影响(`partial=False` 时 `"108."` 仍正常解析为 108.0)。

## 根因
PR 正文披露的根因:partial/streaming 模式下,以 `.` 结尾的裸数值被逗号终结时,`float("108.")` 立即求值为 `108.0`,JSON 序列化后的 `"108.0"` 被发给客户端;当真正的十进制位到达、值变为 `108.2` 时,流式 diff 逻辑计算出公共前缀 `"108."`,但客户端已经收到 `"108.0"`,于是最终 diff 只追加 `"2"`,拼出 `"108.02"`。

## 修复方案
修改 `vllm/tool_parsers/gemma4_tool_parser.py` 的 `_parse_gemma4_args()` 与 `_parse_gemma4_array()`:在 `partial=True`(流式模式)时,若当前裸值(strip 后)以 `"."` 结尾则扣留(withhold)该值不解析,因为后续可能还有十进制位到达;这与既有的字符串结尾扣留逻辑一致。工具调用结束时的最终非 partial 解析能正确处理这些值。关键 diff:

```diff
+            if partial:
+                raw_val = args_str[val_start:i].strip()
+                if raw_val.endswith("."):
+                    # Trailing dot means decimal digits may still arrive
+                    # (e.g. "108." may become "108.2"). Parsing now would
+                    # yield float("108.") == 108.0, whose json repr "108.0"
+                    # corrupts the streaming diff when the true digit lands.
+                    break
             result[key] = _parse_gemma4_value(args_str[val_start:i])
```

数组解析 `_parse_gemma4_array` 中加入同样的尾点检查。并在 `tests/tool_parsers/test_gemma4_tool_parser.py` 新增回归测试 `test_trailing_dot_float_partial_withheld`(args 与 array 两个版本),验证:尾点值整体扣留、尾点值之前的稳定键/元素保留、非 partial 模式照常解析。

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:仅影响 `gemma4_tool_parser.py` 的 partial(流式)解析路径:以 `.` 结尾的裸值会延迟到下一个分片才产出,流式事件顺序上该值出现得更晚,但不改变最终解析结果;非流式(`partial=False`)路径行为完全不变。数组中尾点元素之前的稳定元素仍会先输出。改动不含运行时其他模块。

## 关联
- Fixes #42047: https://github.com/vllm-project/vllm/issues/42047

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
