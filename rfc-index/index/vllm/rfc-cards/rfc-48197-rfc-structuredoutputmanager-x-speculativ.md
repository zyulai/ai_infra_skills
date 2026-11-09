# RFC #48197: [RFC]: StructuredOutputManager x Speculative Decoding Refactor

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48197](https://github.com/vllm-project/vllm/issues/48197) |
| 作者 | yzong-rh |
| 创建日期 | 2026-07-10 |
| 状态 | ⚪ closed(关闭于 2026-09-22) |
| labels | RFC, speculative-decoding, structured-output |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Speculative Decoding |

## 一句话结论
重构结构化输出与投机解码的交互,由 `StructuredOutputManager` 统一负责草稿 token 校验与文法状态推进。

## 提案动机
简化 `Scheduler`、grammar backend 与 `StructuredOutputManager` 之间的契约,尤其在投机解码下:文法非法的草稿 token 需要被拒绝;草稿 token 需临时推进文法状态以生成 bitmask;被拒绝的草稿 token 不应推进文法状态;reasoning 结束边界还可能落在草稿 token 窗口中间,只有边界之后的草稿 token 才应推进文法状态。这让 `Scheduler` 与 `StructuredOutputManager` 需要小心翼翼配合才能保证正确,已产生过实际 bug(如 #34650,最近才由 #44297 修复)。投机解码 x 约束解码至今仍然脆弱或存在隐蔽错误。

## 提出的方案
- 删除 `StructuredOutputManager.should_advance(request)` 与 `trim_reasoning_for_advance(verified tokens)`,新增 `validate_tokens(draft tokens)` 与 `accept_tokens(verified tokens)`,token 校验与文法推进统一由 `StructuredOutputManager` 处理。
- Scheduler step 内每请求流程变为:`validate_tokens` → `grammar_bitmask` → verifier → `accept_tokens`。
- 内部三个方法共用 `_get_constraint_start()` helper 计算第一个应被文法约束的 token:`validate_tokens()` 保留未约束前缀、只验证受约束后缀;`grammar_bitmask()` 在边界前生成未约束 bitmask、边界后生成受约束 bitmask;`accept_tokens()` 只对受约束后缀推进文法状态。
- 由此得到更简单的心智模型:Scheduler 只负责请求生命周期与 token 传递,`StructuredOutputManager` 负责草稿 token 校验、bitmask 生成与 grammar 后端。
- 改动量约 ~100 insertions、~200 deletions(不含测试更新)。

## 优化点与收益
正文未给出量化性能数字。收益在正确性与可维护性:修正 `should_advance()` 对 STRUCTURAL_TAG 与其他约束类型的差别对待、以及从 `num_output_placeholders` 不可靠推断草稿窗口的问题(正文中这两项已标记由 PR #44993 修复);`validate_tokens()` 只能尽力而为的问题仍靠 bitmask + rejection sampling 兜底。预计可修复 #34650、#43388 一类问题。

## 现状与落地
已关闭(关闭于 2026-09-22)。正文仅以删除线更新了两条问题被 #44993 修复,重构方案本身是否落地需查关联 PR。

## 关联
- [#34650](https://github.com/vllm-project/vllm/issues/34650)(投机解码 x 约束解码的 bug)
- [#44297](https://github.com/vllm-project/vllm/pull/44297)(strict tool calling 场景的修复)
- #44993(修复 #34650 对 JSON/REGEX 等约束类型的问题)
- #43388(同类问题)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
