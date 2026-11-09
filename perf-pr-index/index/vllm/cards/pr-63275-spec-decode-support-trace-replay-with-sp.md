# PR #63275: [Spec Decode] Support trace replay with speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aoshen02 |
| 合并 | 2026-10-29 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, speculative-decoding, ready, mrv2 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/63275 |

## 一句话结论

让 MRv2 trace replay 与 spec decode 共存，好 drafter 下 replay 加速 1.47x–2.03x。

## 动机

MRv2 的 `trace_decode_token_ids`（trace replay）此前与 speculative decoding 互斥（文档 limitations 明确列出）。本 PR 允许两者同时启用：draft 仅在其等于该位置 trace token 时才被接受，验证后的 token 由 trace 覆盖，使 replay 在 decode 路径上发出并打分 trace，同时跳过命中 draft 的 target forward pass。

## 优化点

`RejectionSampler._verify` 照常做 rejection sampling，之后对带 trace 的请求调用 `TraceReplayState.apply_trace_to_verified` 重写 verified tokens：从当前 trace step 起，逐个写入 trace token 直到 draft 与 trace 不匹配、到达 bonus 位置或 trace 结束，并据此设置 `num_sampled`。logprobs、rejection 计数、committed tokens 与 drafter 下一步输入都读改写后的 `sampled`/`num_sampled`，不改动其他路径。hook 是每 verify step 一次 kernel launch，无 host 同步，无 trace 的请求在 kernel 内直接返回。正文还说明 penalties 与非 spec replay 保持一致：verify step 的第 i 行折入 drafts 0..i-1，且仅当这些 draft 全部命中 trace 才发出该行，因此每个发出行的历史都是 trace 前缀。

## 关键改动

```text
vllm/v1/worker/gpu/sample/trace_replay.py: 新增 apply_trace_to_verified 与 _trace_replay_verified_kernel
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py: _verify 后接入 trace 重写
vllm/sampling_params.py: 放开 trace replay 与 spec decode 的参数组合校验
docs/serving/online_serving/trace_replay.md: 移除 spec decode 限制并补充行为说明
tests/v1/worker/test_gpu_trace_replay.py: 新增 TestApplyTraceToVerified
```

```python
+    step = tl.load(total_len_ptr + req_state_idx) - tl.load(
+        prompt_len_ptr + req_state_idx
+    )
+    start = tl.load(cu_num_logits_ptr + req_idx)
+    num_draft_tokens = tl.load(cu_num_logits_ptr + req_idx + 1) - start - 1
+    num_replayed = tl.minimum(num_draft_tokens + 1, trace_len - step)
+    matching = tl.full((), 1, tl.int1)
+    for i in range(num_replayed):
```

## 性能声明

正文 Claims：replay speedup with a good drafter **1.47x–2.03x（EAGLE3, Qwen3-8B）**、**1.64x–1.88x（MTP, Qwen3.5-35B-A3B-FP8）**。表格条件：GB300、单 GPU、nightly `8e2beaa3bf` 加补丁、MRv2、64 traces/模型、最长 512 tokens、3/4 temperature 1.0 带 seed + 1/4 greedy、k=3、`logprobs=5`。Qwen3-8B/EAGLE3（RedHatAI/Qwen3-8B-speculator.eagle3）A vs B 1.47x（acceptance 46%）、A1 vs B1 2.03x、Abi vs Bbi 1.82x；Qwen3.5-35B-A3B-FP8/MTP A vs B 1.64x（66%）、A1 vs B1 1.88x；Qwen3-0.6B/ngram A vs B 0.82x（drafter 质量差，更慢）。另在 temperature=0.5/top_p=0.95 下 model-generated A vs B 2.32x、random traces 0.68x。

## 验证与回归风险

新增/改动测试：`tests/v1/worker/test_gpu_trace_replay.py`（`TestApplyTraceToVerified`）、`tests/v1/sample/test_trace_replay_params.py`、`tests/v1/sample/test_logprobs.py::test_trace_replay_with_spec_decode`；旧镜像 1122 passed/18 failed → 新 1140 passed/16 failed，16 个失败为 old/new 共有的环境或基线失败。风险面：bitwise 相等需 `VLLM_BATCH_INVARIANT=1`，processed logprobs 受 top-p masking 在 64 行（`_SPLIT_MAX_BATCH`）的路径切换影响；FP8 MoE 模型本身非 batch invariant；正文明确未覆盖 preemption/resume mid-trace、prefix caching、TP/DP、non-neutral penalties。

## 关联

无（PR 正文未列关联 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
