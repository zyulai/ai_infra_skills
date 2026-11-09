# PR #52436: [Bugfix][Spec Decode][Structured Output] DSpark: fix the grammar bitmask mapping when the draft budget is zero

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52436](https://github.com/vllm-project/vllm/pull/52436) |
| 作者 | oops-oom |
| 合并日期 | 2026-08-16 |
| labels | bug, mrv2, ready, speculative-decoding |
| 热度 | 👍 2 · 💬 30 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
DSpark 自适应验证 budget=0 时 cu_num_logits 被重写,grammar mapping 行数少于 bitmask 行数而断言崩溃;改按 scheduled 布局建 mapping。

## 问题现象
`enable_adaptive_verification`(#47808 引入)与 structured outputs(JSON schema/grammar)叠加时,一旦 drafter 置信度低到自适应验证决定本步验证 0 个 draft,即 crash:`assert num_masks == len(mapping)`。正文强调 budget=0 并非罕见角落:控制器每步按 `argmax_b (est_accepted_tokens(b)/cost(b))` 选 budget,验证任何 draft 都不划算时 0 就是常规输出(高熵 token、预热、grammar 约束步都会触发),持续负载下反复复现。

## 根因
正文披露:scheduler 按 scheduled drafts 定 bitmask 行数(每请求 `len(drafts) + 1` 行),不知道设备侧的 budget 裁剪;而 `compact_batch()` 的 zero-budget 分支把 `cu_num_logits_np` 重写为 bonus-only 布局(该重写服务于 `_iter_request_chunks` 的切片)。`apply_grammar_bitmask` 从重写后的 offsets 推 mapping,每请求只得 1 行而非每 scheduled draft 一行,于是 `assert num_masks == len(mapping)` 失败(2 请求 × 2 draft:bitmask 6 行对 mapping 2 行)。另两个 budget regime 不重写该字段,旧代码只是碰巧带着 scheduled 语义而正确。

## 修复方案
mapping 改由 `num_draft_tokens_per_req + num_bonus_tokens` 计算——该数组来自 scheduler 使用的同一 `scheduled_spec_decode_tokens`(model_runner.py)且不被 compaction 触碰,三个 regime 语义一致;逻辑提取为 `_build_grammar_mapping()`,`model_runner.py` 构造时传入 `num_bonus_tokens`。compacted 设备布局放不下的行由 kernel 内 `position_is_active` mask:

```diff
+        else:
+            # Grammar masks follow the scheduled layout even when adaptive
+            # verification compacts the actual CPU logit offsets to bonus-only.
+            num_positions = int(num_draft_tokens_per_req[req_idx]) + num_bonus_tokens
```

## 影响与回归风险
- 影响窗口:自 #47808 引入 `enable_adaptive_verification` 起(正文说明)。
- 回归面:`structured_outputs.py` 的 grammar bitmask→logits mapping 构造(三个 budget regime 统一为 scheduled 布局语义)、`model_runner.py` 初始化传参;新增测试 `test_zero_budget_keeps_one_grammar_row_per_scheduled_draft` 及关联 chunking/batch-ordering 套件验证。

## 关联
- #47808(引入自适应验证,正文多次引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
