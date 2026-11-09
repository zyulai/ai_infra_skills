# PR #24965: [Spec] Fix ngram metric off-by-1 in `num_accepted_drafts_per_req_cpu`

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24965](https://github.com/sgl-project/sglang/pull/24965) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-05-11 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论

ngram 把 drafts-only 计数当 accept_lens 上报,下游再减 1 导致指标少 1;改传含 bonus 的 num_accepted_tokens。

## 问题现象

ngram speculative decoding 的 per-request 指标 `num_accepted_drafts_per_req_cpu`(及 `spec_accepted_drafts` / `spec_acceptance_histogram`)系统性少 1:下游按"含 bonus"语义再减 1,而 ngram 上游传的已是 drafts-only 计数,得到 `drafts_only - 1` 的错误统计。生成结果本身不受影响。

## 根因

正文披露:ngram 路径把 drafts-only 的 `verify_input.num_accepted_drafts` 作为 `accept_lens` 传出;下游 `scheduler_output_processor_mixin` 期望的是"含 bonus"(即含最终接受的那个 token)的接受长度并自行减 1,于是产出 `drafts_only - 1`。而 `num_accepted_tokens` 在 `ngram_info.py:456` 已按 `num_accepted_drafts + 1` 计算好,正是下游期望的语义。

## 修复方案

`python/sglang/srt/speculative/ngram_worker.py` 的 `forward_batch_generation` 中,把 `accept_lens = verify_input.num_accepted_drafts` 改为 `accept_lens = verify_input.num_accepted_tokens`,并加注释说明 accept_lens 需含 bonus、下游会减 1 还原 drafts-only 计数。

关键 diff 摘录:

```diff
-            # Store accept_lens for per-request metrics
-            accept_lens = verify_input.num_accepted_drafts
+            # Store accept_lens (with bonus) for per-request metrics; downstream
+            # subtracts 1 to recover drafts-only counts.
+            accept_lens = verify_input.num_accepted_tokens
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 ngram worker 存给 per-request 指标的 accept_lens 取值;正文明确说明对生成正确性与 `spec_accept_length` 无影响,只修正 `spec_accepted_drafts` / `spec_acceptance_histogram` 等指标上报。其他 speculative 实现(EAGLE 等)未改动。正文未提供测试说明(无 Accuracy Tests 章节,Checklist 未出现在正文)。

## 关联

正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
