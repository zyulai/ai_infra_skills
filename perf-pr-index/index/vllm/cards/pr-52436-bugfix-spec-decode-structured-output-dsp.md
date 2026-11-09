# PR #52436: [Bugfix][Spec Decode][Structured Output] DSpark: fix the grammar bitmask mapping when the draft budget is zero

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @oops-oom |
| 合并 | 2026-08-16 |
| 分类 | Other (tier: major) |
| Label | bug, speculative-decoding, ready, mrv2 |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/52436 |

## 一句话结论

修复 DSpark 自适应验证 draft budget 为 0 时结构化输出 grammar bitmask 映射行数错配导致的 assert 崩溃。

## 动机

#47808 引入的 DSpark 自适应验证(`enable_adaptive_verification`)与 structured outputs(JSON schema / grammar)组合时,只要 drafter 置信度低到 controller 选取 `draft_budget = 0`,服务即崩溃:`assert num_masks == len(mapping)` 失败。draft budget 为 0 并非罕见边角:controller 每步在 `b ∈ [0, max_draft_budget]` 上做 `argmax_b (est_accepted_tokens(b)/cost(b))`(`adaptive_verification.py:329`),当连最自信的 draft 的验证成本都为负(高熵 token、warm-up、grammar 约束步)时 b=0 是常规输出,因此在普通负载下可稳定复现。

## 优化点

根因是 scheduler 与 worker 对 grammar bitmask 行数语义不一致:scheduler 按 scheduled drafts 每请求分配 `len(drafts) + 1` 行(`vllm/v1/structured_output/__init__.py`),不知道设备侧的 budget 裁剪;而 `compact_batch()` 的零预算分支会把 `cu_num_logits_np` 改写成 bonus-only 布局,`apply_grammar_bitmask` 原先从这份被改写的偏移推导 bitmask→logits 映射,结果每请求只剩 1 行,触发 assert。修复是把映射行数改由 `num_draft_tokens_per_req + num_bonus_tokens` 决定——该数组与 scheduler 同源(来自 `scheduled_spec_decode_tokens`,`model_runner.py:1132`)且不被 compaction 触碰,三种 budget 状态下行数语义一致。被压缩后设备布局放不下的行,内核里已由 `position_is_active`(`structured_outputs.py:144`)按 GPU 侧 `cu_num_logits` 屏蔽,host 侧不再二次猜测。只有 budget==0 分支会改写 `cu_num_logits_np`,另两个 regime(`budget == num_drafts`、`0 < budget < num_drafts`)原代码只是碰巧正确。

## 关键改动

```text
vllm/v1/worker/gpu/structured_outputs.py: 抽出 _build_grammar_mapping,行数改按 num_draft_tokens_per_req + num_bonus_tokens 计算
vllm/v1/worker/gpu/model_runner.py: 向 StructuredOutputsWorker 传入 num_bonus_tokens
tests/v1/spec_decode/test_adaptive_verification.py: 新增 test_zero_budget_keeps_one_grammar_row_per_scheduled_draft
```

关键摘录(_build_grammar_mapping 的核心分支):

```python
if num_draft_tokens_per_req is None:
    num_positions = int(
        cu_num_logits_np[req_idx + 1] - cu_num_logits_np[req_idx]
    )
else:
    # Grammar masks follow the scheduled layout even when adaptive
    # verification compacts the actual CPU logit offsets to bonus-only.
    num_positions = int(num_draft_tokens_per_req[req_idx]) + num_bonus_tokens
```

## 性能声明

无性能数字声明;本 PR 为正确性修复。PR 未注明 benchmark 数据。

## 验证与回归风险

新增回归测试 `test_zero_budget_keeps_one_grammar_row_per_scheduled_draft`,并验证其在不带 fix 时确实失败(`assert 3 == 7`,即旧映射 len 3 对 bitmask 行数 7)。测试结果:本套件 6 passed;相关共用 `cu_num_logits_np` 契约的 `test_gpu_rejection_sampler_chunking.py`、`test_gpu_batch_ordering.py` 共 17 passed / 4 skipped;pre-commit(含 mypy、ruff format)全部通过。影响路径:GPU worker 的结构化输出 + spec decode 组合;`num_draft_tokens_per_req is None` 分支保留原 cu_num_logits 行为,非自适应验证路径不受影响。

## 关联

依赖 #47808(DSpark 自适应验证)引入的行为;同区域为 spec decode + structured output 交叉路径。本卡所在批次素材中未列出其他同区域 PR,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
