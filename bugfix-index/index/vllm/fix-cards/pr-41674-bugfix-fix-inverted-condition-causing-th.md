# PR #41674: [Bugfix] Fix inverted condition causing thinking_token_budget to be silently ignored

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41674](https://github.com/vllm-project/vllm/pull/41674) |
| 作者 | JasonKeyiL |
| 合并日期 | 2026-05-15 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
gpu_input_batch.py 中 `or not thinking_budget_tracks_reqs` 条件写反;删除多余的 not,修正 budget 跟踪请求拿不到 output_token_ids 的潜在错误。

## 问题现象
正文说明:该反转条件在逻辑上把语义弄反——批量中没有 budget-tracking 请求时反而把 `needs_output_token_ids` 置 True,有时反而贡献 False。实际当前被掩盖:同一 commit 引入的 init 逻辑在 `reasoning_config is not None` 时把 `logitsprocs_need_output_token_ids` 置 True,`output_token_ids` 永不为空,budget 端到端仍正确工作;此修复纠正潜在逻辑错误,并免去批内无 budget-tracking 请求时不必要的 `output_token_ids` 填充。

## 根因
正文披露:`gpu_input_batch.py` 中布尔条件反转(`or not thinking_budget_tracks_reqs` 应为 `or thinking_budget_tracks_reqs`)。

## 修复方案
删除错误的 `not`(单 token 改动);新增回归测试 `test_thinking_budget_enforced_without_penalties`:直接驱动 `ThinkingBudgetStateHolder`,验证传空列表(修复前行为)时 `_update_think_state` 被跳过、in_end 不置位,传真实 output_token_ids 且思考 token 达到预算后 in_end 置位:

```diff
             or self.logitsprocs_need_output_token_ids
-            or not thinking_budget_tracks_reqs
+            or thinking_budget_tracks_reqs
```

## 影响与回归风险
- 影响窗口:正文说明该条件与 `logitsprocs_need_output_token_ids` 的 init 逻辑在同一 commit 引入,未给出具体版本号。
- 回归面:`gpu_input_batch.py` `_make_sampling_metadata` 的 `needs_output_token_ids` 判定:有 budget-tracking 请求时 `output_token_ids` 会被填充(budget 生效所依赖);无 budget-tracking 请求时不再填充,减少无谓开销。端到端行为在当前被掩盖的状态下不变。

## 关联
- Fixes #41672

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
