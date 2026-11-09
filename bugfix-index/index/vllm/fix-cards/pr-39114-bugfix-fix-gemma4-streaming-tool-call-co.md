# PR #39114: [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39114](https://github.com/vllm-project/vllm/pull/39114) |
| 作者 | sfeng33 |
| 合并日期 | 2026-04-08 |
| labels | bug, ready, tool-calling |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

Gemma4 流式工具调用中,被 chunk 切开的布尔/数字字面量被当成裸字符串,类型漂移毁掉 JSON;partial 模式下扣留末尾不完整值。

## 问题现象

以 `vllm serve google/gemma-4-31B-it --enable-auto-tool-choice --tool-call-parser gemma4` 启动并发送流式(stream=true)chat completions 请求、让模型调用带 boolean 参数的工具时,若布尔/数字值恰好在 token 边界被切成多个 chunk,流式增量会损坏。作者的手动复现显示:修复前流式 chunk 产出 `"search_all": "trutrue`(已损坏,字面量片段拼接成字符串);修复后产出 `"search_all": true`(正确的布尔值)。触发条件是布尔/数字值跨 chunk 分裂的流式工具调用;非流式路径不受影响(正文只提 streaming)。

## 根因

正文明确披露:`_parse_gemma4_value()` 把"不完整的字面量"(如布尔 `true` 的前缀 `tru`)误判为裸字符串(bare string)。于是流式过程中同一个值的类型在 string 与 boolean 之间漂移(type change,string → boolean),基于参数 diff 的增量输出把这些前后不一致的片段拼接进 JSON,造成流式 JSON 损坏。

## 修复方案

改 `vllm/tool_parsers/gemma4_tool_parser.py`,给解析函数引入 `partial` 标志:

1. `_parse_gemma4_args` / `_parse_gemma4_array` 新增关键字参数 `partial: bool = False`;partial 模式下,扫描到字符串末尾仍未遇分隔符的裸值视为可能不完整、类型不稳定(如半个布尔),直接 `break` 扣留不输出;未闭合的嵌套对象/数组改用 `i`(而非 `i-1`)切分并以 `partial=True` 递归:

```diff
+            if partial and i >= n:
+                # Value may be incomplete (e.g. partial boolean) —
+                # withhold to avoid type instability during streaming.
+                break
             result[key] = _parse_gemma4_value(args_str[val_start:i])
```

2. `_emit_argument_diff` 把流式解析入口从 `_parse_gemma4_args(raw_args_str)` 改为 `_parse_gemma4_args(raw_args_str, partial=True)`;非流式的完整解析仍走默认 `partial=False`,行为不变。
3. `tests/tool_parsers/test_gemma4_tool_parser.py` 新增 3 个用例:`true` 切成 `tru`+`e`、`false` 切成 `fals`+`e`、数字 `42` 切成 `4`+`2`,断言流式汇总后类型正确。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,改动只影响 Gemma4 工具解析器:流式路径(`_emit_argument_diff`)每个增量中位于字符串末尾的裸值会被扣留一轮,下一次 chunk 到达时作为完整值输出,即流式中该参数的可见性延后一步;嵌套对象/数组的未闭合部分切片方式变化(`obj_start:i`)。非流式与完整解析路径(`partial=False`)保持原行为。其他模型的 tool parser 不受影响。

## 关联

- https://github.com/vllm-project/vllm/issues/39089 (Fix)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
