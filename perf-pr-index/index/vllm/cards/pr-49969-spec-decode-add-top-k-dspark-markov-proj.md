# PR #49969: [Spec Decode] Add top-k DSpark Markov projection

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @askliar |
| 合并 | 2026-08-04 |
| 分类 | Speculative Decoding \| tier: major |
| Label | documentation, speculative-decoding, ready, v1, qwen, mrv2 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/49969 |

## 一句话结论

DSpark 只对 top-k 候选做 Markov 投影，GB10 并发 64 下 per-user 吞吐 54.0 → 78.1 tok/s（+45%）。

## 动机

PR 正文 Purpose：DSpark 先并行计算所有 draft 位置的 base logits，再在整个 draft vocabulary 上顺序施加 Markov bias；这段重复的 full-vocabulary Markov projection 处在串行的 drafting 路径上，是待优化点。

## 优化点

新增面向 Qwen3 DSpark 的 `dspark_draft_topk`：从 base logits 一次性选出 top-k 候选，gather 对应的 `markov_w2` 行，只对这批候选施加顺序 Markov bias。当前实现刻意复用已有 dense sampling path：计算 dense base logits 并对所有 draft 位置选一次 top-k；把可复用的 dense logits buffer 初始化为 `-inf`；每个 sequential step 仅对选中候选计算 Markov bias 并 scatter 回 dense 行；现有 greedy 或 probabilistic Gumbel/rejection sampling 路径保持不变。Base LM-head 输出与 probabilistic draft-logits 存储仍为 dense，该优化只减少重复的 sequential Markov projection，避免自定义 proposal 与 rejection-sampling plumbing。Current scope 为 Qwen3 DSpark。改动落在 `vllm/config/speculative.py`（`dspark_draft_topk` 字段及校验，要求 method=dspark、draft TP size 1）、`vllm/model_executor/models/qwen3_dspark.py`（`apply_bias_gathered`）、`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` 与新增测试。

## 关键改动

```text
vllm/config/speculative.py: 新增 dspark_draft_topk 字段与校验（仅 DSpark、draft TP size 1）
vllm/model_executor/models/qwen3_dspark.py: DSparkMarkovHead.apply_bias_gathered 对 top-k 候选施加 Markov bias
vllm/v1/worker/gpu/spec_decode/dspark/speculator.py: drafter 接入 top-k gather 路径
tests/v1/spec_decode/test_dspark_topk.py: 新增 gathered bias 与 dense 一致性测试
```

```text
+    dspark_draft_topk: int | None = Field(default=None, ge=1)
+    """For Qwen3 DSpark drafting, evaluate the Markov projection only for the
+    top-k base-logit candidates. Requires draft tensor parallel size 1."""
...
+    logits.fill_(float("-inf"))
+    result = _markov_head(weight).apply_bias_gathered(
+        markov_embed, logits, values, index
+    )
```

## 性能声明

End-to-end（`draft_sample_method: probabilistic`、temperature 1、7 speculative tokens、GB10；表格注 ¹ Concurrency 64、ISL 32768 / OSL 1024）：Initial PR top-k 512 GSM8K strict/flex 0.82/0.83、AIPerf tok/s/user 65.3、total 44.6；Initial Off 0.80/0.83、53.0、40.6；Current simplified top-k 512 0.84/0.84、78.1、52.1；Current Off 0.84/0.85、54.0、41.3。正文："At concurrency 64, top-k 512 improves per-user throughput by 23% for the initial implementation (53.0 → 65.3) and 45% for the current simplified implementation (54.0 → 78.1)"；"With top-k 512, the current implementation is 20% faster per user than the initial implementation (78.1 versus 65.3)"；GSM8K 分数在 run-to-run 范围内，无实测精度回退。Decode GPU profile（concurrency-1 / 约 8K-context decode window，毫秒）：Drafter projection (cutlass/GEMV) Initial/Off 155、Initial/512 105、Current/Off 154、Current/512 107；Total GPU time 878.5 / 822.9 / 870.8 / 831.4。正文："Top-k 512 reduces the drafter projection by approximately 47–50 ms. The top-k selection itself adds only approximately 1 ms. Total decode GPU time falls by 6.3% for the initial implementation and 4.5% for the current implementation." Acceptance-length sweep：t=0 greedy 从 top-k512 的 3.9509 到 Off 的 3.9253，t=1 probabilistic 从 3.6582 到 3.6413。Benchmark caveats：AIPerf 用 5 measured + 2 warmup requests、AIPerf 与 GSM8K `gpu-memory-utilization=0.8`、Nsight 用 0.7（0.8 下 CUPTI 导致 model-load OOM）。

## 验证与回归风险

Focused checks：`pre-commit run --files <changed files>` passed（含 Ruff、mypy）；`pytest -q tests/v1/spec_decode/test_dspark_topk.py` 2 passed；`pytest -q tests/test_config.py -k 'draft_sample_method'` 2 passed；`pytest -q tests/v1/worker/test_gpu_rejection_sampler_chunking.py -k 'preserves_request_boundaries'` 1 passed。Test Plan 含 dense vs top-k 的低/高并发对比、GSM8K 与 acceptance length、decode window profile。回归风险：scope 仅 Qwen3 DSpark，配置校验要求 method=dspark 且 draft TP size 1；正文称 MoE routing/GEMMs、dense matmuls、attention 等主要 target-model 类别在 run-to-run 变化内不变；AIPerf 样本少（5 measured + 2 warmup），吞吐值存在 run-to-run variance。Test Plan 还要求对比 dense DSpark 与 dspark_draft_topk 在低/高并发下的表现、检查 GSM8K 与 acceptance length、profile decode window 以确认节省发生在 drafter projection，并运行 focused 配置/top-k 测试与仓库 pre-commit hooks。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
