# PR #46779: [CI/Build] Fix incorrect CLI arg validation tests in test_cli_args.py

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46779](https://github.com/vllm-project/vllm/pull/46779) |
| 作者 | sungbin1015 |
| 合并日期 | 2026-10-06 |
| labels | ready, agents |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
两个 CLI 参数校验测试断言了错误契约、给出虚假信心;改为验证文档化行为(reasoning-parser 与 auto tool choice 可组合),纯测试改动。

## 问题现象
正文描述:`tests/entrypoints/openai/test_cli_args.py` 中两个测试给出虚假信心——(1) `test_enable_auto_choice_fails_with_enable_reasoning` 断言 `--enable-auto-tool-choice` 与 `--reasoning-parser` 组合必须失败,但该组合在 `docs/features/tool_calling.md` 中被文档化为支持(如 `--tool-call-parser hunyuan_a13b --reasoning-parser hunyuan_a13b`),`validate_parsed_serve_args` 也正确地不拒绝它;旧测试之所以"通过",只是因为漏了 `--tool-call-parser`,实际触发的是无关的"auto tool choice 需要 tool call parser"检查,掩盖了根本不存在(也不应存在)reasoning 冲突校验这一事实。(2) `test_enable_auto_choice_passes_without_tool_call_parser` 名不副实:docstring 与函数体断言的都是校验失败。

## 根因
正文披露:测试编写时对 `validate_parsed_serve_args` 的契约理解错误——把"漏配 tool-call-parser 会失败"误当成"与 reasoning-parser 组合会失败";纯测试缺陷,运行时行为无错误。

## 修复方案
- 测试 1 重写为 `test_auto_tool_choice_with_reasoning_parser_passes`:补上 `--tool-call-parser mistral` 后断言 `validate_parsed_serve_args` 通过,docstring 引用 tool_calling.md 说明该组合必须通过。
- 测试 2 更名为 `test_enable_auto_choice_fails_without_tool_call_parser`,使名称与实际断言(无 parser 时失败)一致。
- 纯测试改动,不修改运行时行为(diff 文件为 `tests/entrypoints/launchers/test_cli_args.py`)。正文注明已检索开放 PR,非重复改动;结果 28 passed,ruff 通过。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围。
- 回归面:仅 CI 测试文件;`validate_parsed_serve_args` 与 CLI 行为不变。风险是未来有人基于旧测试的(错误)预期给校验器加 reasoning 冲突逻辑——本修复使测试契约与文档一致。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
