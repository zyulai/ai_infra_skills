# PR #38172: [Misc] Add 20 regression tests for 11 tool parser bug fixes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38172](https://github.com/vllm-project/vllm/pull/38172) |
| 作者 | bbrowning |
| 合并日期 | 2026-04-01 |
| labels | bug, deepseek, qwen, ready, tool-calling |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
为 2025-09 以来 11 个缺测试覆盖的工具解析器修复补 20 个回归测试,防止重构时回归。

## 问题现象
本 PR 是纯测试补充("purely additive test coverage"),不修新 bug。问题现象是过程性的:作者与 Claude Code 审计了 2025 年 9 月至今的工具解析器 bug-fix PR,发现其中多个在合入时没有对应的测试覆盖;在后续对这些区域进行 refactor、cleanup、redesign 时,这些修复存在被无意回归的风险。被覆盖的修复包括:Mistral 快速去分词下的文本检测(#37209);Qwen3Coder 畸形 XML 崩溃、anyOf 双重编码、投机解码流式(#36774、#36032、#35615);DeepSeekV32 快速去分词下保留分隔符与 skip_special_tokens 调整(#33964);GLM-4 MoE 零参数工具调用、transformers 5.x 分隔符处理、Unicode 字符保留(#32321、#31622、#30920);MiniMax M2 anyOf 可空参数(#32342);Step3p5 MTP 式变长分块与多 token 流式(#33690);Kimi K2 原生 tool call ID 提取与多轮 ID 连续性(#32768)。

## 根因
正文未披露新的运行时根因——本 PR 不含产品代码改动。其动因为正文所述:上述 11 个修复合入时缺少回归测试。每个测试的 docstring 标注了对应原 PR 的缺陷机制,例如「fast detokenization skipping DSML delimiters (PR #33964)」、「bot_token in text but not token_ids (PR #37209)」等。

## 修复方案
在 `tests/tool_parsers/` 下 7 个测试文件新增约 20 个回归测试,全部只动测试代码:
- `test_deepseekv32_tool_parser.py`:新增 `TestDelimiterPreservation`,覆盖 DSML 分隔符以普通文本出现时仍被识别、`adjust_request` 置 `skip_special_tokens=False`,并为 mock tokenizer 补 `tokenize` 返回值。
- `test_glm4_moe_tool_parser.py`:零参数工具调用、畸形 tool_call 无 regex 匹配、transformers 5.x 分隔符处理、Unicode 不被转义为 `\uXXXX`。
- `test_kimi_k2_tool_parser.py`:原生 ID(`functions.get_weather:0`)落到 ToolCall 上、跨轮次新 parser 的 ID 连续性。
- `test_minimax_m2_tool_parser.py`:`anyOf` 可空参数的非空串、null、object 三种取值。
- `test_mistral_tool_parser.py`:bot_token 出现在文本但不在 token_ids 时的检测(含 pre-v11 tokenizer)。
- `test_qwen3coder_tool_parser.py`:无 `>` 的畸形 XML 不崩溃、None tool call 被过滤、anyOf 不双编码、单 chunk 多参数、字符串参数不双重序列化。
- `test_step3p5_tool_parser.py`:MTP 变长 chunk 跨参数边界、大 chunk 跨多个 tool call 并与逐 token 流式结果对齐。

## 影响与回归风险
- 影响窗口:不适用(纯测试改动,无运行时行为变化)。
- 回归面:仅影响 `tests/tool_parsers` 测试套件;正文 Test Plan 为 `pytest -sv tests/tool_parsers`,结果 `510 passed, 1 skipped, 42 xfailed, 2 warnings`,新测试全过且旧测试不受影响。对生产路径零风险,但对上述解析器的后续重构起到护栏作用。

## 关联
- vllm-project/vllm#37209、vllm-project/vllm#36774、vllm-project/vllm#36032、vllm-project/vllm#35615、vllm-project/vllm#33964
- vllm-project/vllm#32321、vllm-project/vllm#31622、vllm-project/vllm#30920、vllm-project/vllm#32342、vllm-project/vllm#33690、vllm-project/vllm#32768

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
