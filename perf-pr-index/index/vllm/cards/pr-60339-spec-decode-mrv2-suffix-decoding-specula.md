# PR #60339: [Spec Decode][MRV2] Suffix decoding speculator for Model Runner V2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @nishantrevur |
| 合并 | 2026-11-03 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | documentation, speculative-decoding, ready, mrv2, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 8 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/60339 |

## 一句话结论

suffix 投机解码迁到 MRV2 的 GPU speculator，接受率 1.937 tok/step，与 Arctic 1.923 相当。

## 动机

正文指出 `method="suffix"` 此前依赖 Arctic Inference、跑在 CPU 上、强制 Model Runner V1 且禁用 async scheduling。本 PR 让它在 Model Runner V2 上以 GPU speculator 运行（GPU-resident、propose path 无 host sync、async scheduling、无 Arctic 依赖），V1 仍用 Arctic。这是 #47172 的 "suffix" 项；speculator 基于 ngram_gpu(#40704) 并借鉴 #52097。

## 优化点

机制：Own history 内最长后缀匹配（同 ngram_gpu kernel layout）；Cross-request corpus 用 GPU token ring（`suffix_decoding_corpus_tokens` 默认 1M），对每请求 last 2 / 4 tokens 做 hash index 返回最多 16 个候选位置，逐个验证并扩展到完整后缀匹配；Frequency vote 按候选匹配长度对续接投票，最高分链胜出、平局取更长；Adaptive verification 按 slot 置信度在线学习裁剪不值得验证的 slot；Ingestion 每 (document, block) 一个 GPU program，精确且顺序无关；TP 一致性无需 collective（finished ids 排序、index 用 atomic_max、统计为整型计数器）。默认内存 36 MB（4 MB corpus + 32 MB index）。`num_speculative_tokens` 默认改为 8（原 `suffix_decoding_max_tree_depth` 24），因 V2 验证每个 slot；BaseSpeculator 新增 add_request / on_requests_finished / reset_state 三个 no-op hook。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/suffix/__init__.py、config.py、vllm/v1/spec_decode/suffix_decoding.py: V2 GPU speculator 与 V1 Arctic 依赖检查
vllm/v1/worker/gpu/spec_decode/speculator.py、speculator_config.py: BaseSpeculator hooks 与配置
vllm/v1/worker/gpu/model_runner.py: finish_requests / add_requests 调用 speculator hooks
vllm/config/speculative.py、docs/features/speculative_decoding/*: 配置与 suffix / adaptive verification 文档
tests/v1/spec_decode/test_suffix_speculator.py、test_custom_speculator.py 等: 单测与 e2e
```

```python
    def add_request(self, req_index: int) -> None:
        """A new request was assigned request-state slot `req_index`, e.g. to
        reset per-slot state. No-op by default."""
        return

    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        """Requests finished (not preempted), before their request-state slots
        are freed so their token history is still readable. No-op by default."""
        return
```

## 性能声明

Acceptance：21,299 条 greedy Qwen2.5-1.5B 请求（5.1M output tokens，UltraChat/Alpaca/GSM8K/Spec-Bench/code edit/rewrite/templated JSON，waves of 64，1M-token corpus，k=8）上 1.937 tokens/step，对比 Arctic(V1) k=8 的 1.923 与 k=24 的 1.948。Throughput（offline LLM.generate，output tok/s vs no speculation）：Qwen2.5-7B concurrency 1/8/32/64 时本 PR 1.46/1.32/0.82/0.59x、+adaptive verification 1.39/1.36/1.24/1.18x（Arctic 1.31/1.29/1.24/1.03x）；Qwen2.5-1.5B 1/8/32/64/128/256 时本 PR 1.66/1.47/0.97/0.64/0.56/0.52x、+adaptive 1.56/1.29/1.15/1.05/0.99/0.91x。Online serving（vllm bench serve，Qwen2.5-7B，c1/8/32/64）：本 PR 1.49/1.35/0.87/0.65x、+adaptive 1.44/1.44/1.30/1.23x。GSM8K（Qwen2.5-7B-Instruct 1319 题 5-shot）：no spec 83.0%、本 PR 82.8%、+adaptive 82.9%，匹配无投机。Hash index propose 成本 11-35 µs GPU（batch 1-256，CUDA-graph 微基准），引擎 eager 跑时 4K context 89-95 µs/step、128K context 32 requests 231 µs。c1 中位 TTFT 125 ms（本 PR）vs 73 ms 无投机、74 ms Arctic V1，约一个 decode step（V2 spec-decode 成本）。测试条件：模型 Qwen2.5-7B / 1.5B / 0.5B、Qwen2.5-VL-3B，硬件 L4（e2e 单卡），精度 greedy / fp32 / bf16，workload 离线 LLM.generate、vllm bench serve 与 21K-request replay。正文注明结果测于 @njhill 的 rebase/简化之前（该改动只影响带 explicit seed 的请求，benchmark 不发 per-request seeds，故仍适用），#61509 已合并所以 "with #61509" 是当前 main 的数字。

## 验证与回归风险

单测：test_suffix_speculator.py + test_adaptive_verification.py 49 passed、test_config.py -k 相关 68 passed、test_ngram_gpu.py + test_custom_speculator.py 58 passed、test_ngram_suffix.py -k suffix 3 passed；单 L4（Llama-3.1-8B-Instruct）e2e：V2 GSM8K 76.3% 通过（旧默认 k=24 会 OOM）、V2 cross-request 0.74→4.47 accepted tokens、Arctic(V1) acceptance 通过。Runtime checks：VLLM_GPU_SYNC_CHECK=error 无新增 host sync；TP=2 下 312 请求后两 rank 的 corpus/index/statistics/drafts 一致。回归风险：seeded sampling 与 suffix 不可复现（drafts 来自已完成请求）、corpus 跨请求共享忽略 cache_salt / LoRA adapter、adaptive verification 需 full CUDA graphs 与可变长 decode batch 的 attention backend（L4 上 TRITON_ATTN，FA2 仅声明 uniform-batch）、长上下文下其 startup profiling 曾耗 1.9 GiB KV（#61509 修复）、V2 TTFT 比无投机高约一个 decode step；正文另列 5 个 open questions 与 follow-ups（含 aarch64 benchmarks）。

## 关联

#47172（"suffix" 项）、#40704（ngram_gpu 基础）、#52097（借鉴的 suffix_gpu on V1，作者协作）、#61509（adaptive verification 的 startup profiling 计为 CUDA graph 内存的相关修复）、#44056 与 #50518（spec-decode metrics 的 follow-up 参照）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
