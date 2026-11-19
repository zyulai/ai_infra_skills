# PR #63114: [Model] Serve Base Labs mammoth-1 / elephant-1 with their trained block-bidirectional mask

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simon-veitner-redhat |
| 合并 | 2026-11-17 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | documentation, new-model, frontend, ready, ci/build, qwen, mrv2, rust, scheduler, kv-cache-manager |
| Diff | pack 标注 +None/-None, None files（正文自述 21 files，+1632/−10，tests +961/−1）；pack 末尾说明「13 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/63114 |

## 一句话结论

为 Base Labs 决策模型 mammoth-1 (Qwen3.5-27B) 与 elephant-1 (9B) 按训练时的 block-bidirectional mask 提供服务（stock vLLM 会忽略该 mask），JevBench 精度提升且 prefix caching 保持开启，DI 0.3 样本 c64 下 elephant-1 提速 1.58x、mammoth-1 提速 1.48x。

## 动机

Base Labs 的 mammoth-1 与 elephant-1 在 config.json 的 `modeling_overrides` 中声明了 full-attention 层的 block-bidirectional mask（块内 query 可见本块后继 keys，块间保持因果），stock vLLM 忽略该 mask 导致以错误方式读题。本 PR 让模型按训练方式在 `/v1/systemone` 与 `/v1/decisions` 上服务，同时保留 prefix caching。约束：prefix token 的 KV 与 Mamba state 依赖整个 prefix，question token 依赖整个 prompt，故 cache hit 只在 P 以内且同一 prefix 下有效，且一步绝不能在 block 内部结束。

## 优化点

- **`PrefixLMScheduler`**（scheduler 子类 + `PrefixLMAsyncScheduler` 孪生）：hit 截止于 `min(P, num_tokens - 1)` 并遵守 upstream `max_cache_hit_length`；hit 报告无 shared-prefix junction（0）以免 Mamba split 在 prefix 内切步；prompt 其余部分经 stock `Scheduler` 一步 prefill 完（chunked prefill 关）或在队头等待；缺盐/伪造盐的请求移入独立 namespace、causal 请求移出；流式与多模态请求按因果读取（一次性 warning）。实现方式为覆写一个私有方法 `_get_local_prefix_cache_hit`。
- **`Qwen3_5ForSlotDecision` + `SlotLMStrategy` + `Qwen3_5SlotDecisionModelState`**：模型状态类覆写既有 `prepare_attn`（如同 `diffusion_gemma.py` 与 DeepSeek V4.1），prefill 步上构造 ranges `[(0, P-1), (P, L-1)]`，super() 构建 GDN groups、`build_attn_metadata(mm_req_doc_ranges=ranges)` 构建 full-attention groups；不写任何已构建 metadata 字段、不 import backend 类。步在 block 内结束会 raise 而非返回错误读数；FULL CUDA graph 的单 token query 跳过 ranges。
- **两个核心编辑**：删除 `MambaModelConfig` 的一个 assert（让所有 hybrid 模型可组合 Mamba prefix caching 与 `--no-enable-chunked-prefill`）；`MAX_LOGPROB_TOKEN_IDS` 128 → 256（Python 与 Rust）。
- **模型默认设置**：`--max-model-len` 缺省为训练长度 16,384（显式值保留）；`--max-num-batched-tokens` 低于 max-model-len 时提升到它；chunked prefill 对服务器上所有请求关闭；attention backend 强制 `TRITON_ATTN`（唯一按请求应用 ranges 的后端）；prefix caching 默认开但需 FlashInfer GDN prefill（Hopper，或 CUDA 13 的 Blackwell），否则带日志关闭。
- 第三 commit（by @LucasWilkinson）让 PrefixLM prompt 走 stock scheduler 路径，去掉子类自己的 Mamba split。

## 关键改动

```text
vllm/v1/core/sched/prefix_lm_scheduler.py（新）: PrefixLMScheduler / PrefixLMAsyncScheduler / PREFIX_LM_LEN / prefix_lm_salt
  （cache_salt = "prefix-lm:" + sha256(prefix); KV-event 消费者可见 "prefix-lm:<digest>[:<user salt>]"）
vllm/models/qwen3_5_slot.py（新）: Qwen3_5ForSlotDecision + Qwen3_5SlotDecisionModelState（_block_ranges、步内切块 raise）
vllm/entrypoints/openai/decisions/slot_prompt.py（新）: SYSTEM_PROMPT / SLOT_LABELS（A–Z + 双字母至 256）/ render_suffix
vllm/entrypoints/openai/decisions/strategies.py: + class SlotLMStrategy(NextTokenStrategy)（fp32 head 校验、n_slots≤256、
  MAX_SCORE_LEVELS=10、MAX_OPTIONS=255、read() 按 extra_args["prefix_lm_len"]=P 发 PrefixLM 请求）
vllm/models/config.py: Qwen3_5ForSlotDecisionConfig hook + 删除 MambaModelConfig 的 assert
vllm/sampling_params.py: MAX_LOGPROB_TOKEN_IDS 128 -> 256
tests/（5 个新文件 + registry/utils）: test_block_bidirectional.py（对照 Transformers block mask，eager 与 CUDA graph）、
  test_slot_decision_e2e.py（RemoteOpenAIServer 走 /v1/systemone 与 /v1/decisions）、test_prefix_lm_scheduler.py（12）、
  test_qwen3_5_slot.py（23-24，EngineArgs(...).create_engine_config() 构真实 VllmConfig）、test_slot_strategy.py（9）
docs/models/structured_decisions_models.md: 新增 Mammoth-1 and Elephant-1 章节
```

## 性能声明

- **JevBench public**（231 tasks，`/v1/systemone`，this head，H200）：mammoth-1 acc 0.9307 → 0.9394、Brier 0.1165 → 0.0833（hard split 0.8559 → 0.8739）；elephant-1 acc 0.8571 → 0.9004、Brier 0.1752 → 0.1296（hard split not split for stock → 0.7928）。
- **Decision Index public 0.3**（earlier head，B200，未在本 head 重跑）：mammoth-1 72.95 → 75.36（board 75.34）；elephant-1 67.89 → 72.27（board 72.24）；全部 140,107 行 ok。
- **吞吐（DI 0.3 样本，362 行 3,214 题，decisions/s，this head，H200，1 warm-up + 3 timed rounds，各 round 新 cache_salt）**：elephant-1 c1 96.07、c16 123.57、c64 124.88、c64 无 prefix caching 78.96（**1.58x**）；mammoth-1 c1 30.60、c16 33.36、c64 33.32、c64 无缓存 22.52（**1.48x**）。每轮 cache on 时 1,718,338 prompt tokens 中 658,944（elephant-1）/ 582,400（mammoth-1）为 prefix hit。mammoth-1 增益较小：attention block 832 tokens，短共享前缀拿不到 hit；且在 H200 上 c1 即 prefill-bound。
- **无共享前缀的代价**（JevBench 231×8=1,848 请求各带独立 salt，0 hit，c64）：elephant-1 68.09 → 55.38 decisions/s（−19%）；mammoth-1 19.87 → 15.78（−21%）。slot prompt 比 stock 长 10%（每轮 1,280,904 vs 1,162,320 prompt tokens），且 `TRITON_ATTN` 替代 FlashAttention。
- **16k 延迟**（16,000-token state + 64 题 4 选项，冷启动 5 次）：elephant-1 2.744 s（2.707–2.848）；mammoth-1 6.715 s（6.694–6.741）。
- **混合流量**（DI 行 c16 + 流式 causal chat）：causal TTFT p50 各行区间与 control 重叠（如 elephant-1 2 workers 0.955 s vs 0.918 s），slot decisions/s 亦重叠，0 preemptions；样本小（每臂 18–36 个 causal 请求）。
- **其他**：强制 TRITON_ATTN 相对 FlashInfer（mask off）损失 2.5–5.4% decisions/s（c1–c64）、decode 0.8–4.3%（earlier head）；Qwen3.5-9B 默认设置 bitwise-equal、19.90 vs 19.93 req/s（−0.15%）；`MAX_LOGPROB_TOKEN_IDS` 翻倍使 `LogprobTokenIdsState` 每 1,024 请求增 0.5 MiB、DiffusionGemma stash 增 48 MiB（1,024 请求、32-token canvas），测得无步耗时（约 11.7 ms/步，K=0/128/255）。

## 验证与回归风险

GPU（H200，前三 commit）：test_block_bidirectional 2、test_slot_decision_e2e 4、test_prefix_lm_scheduler 12、test_qwen3_5_slot 23、test_slot_strategy 9、test_initialization -k Qwen3_5ForSlotDecision 1 全过；core：test_scheduler -k "mamba or chunked" 21 passed、test_mamba_align_chunk_split 48 passed。CPU（this head）：45 passed、6 skipped。pre-commit 通过。`VLLM_GPU_SYNC_CHECK=error` 下 elephant-1 JevBench c8 + DI c16（955 请求）无 sync 报告。质量：JevBench 462/462 bitwise；DI 概率 max |Δp| 0.012/0.015 在同树 run-to-run 噪声（0.015/0.029）内；cached tokens 每读相等（1,186 reads/model）；流式带 `prefix_lm_len` 读数与因果 bitwise 一致。回归风险：删除断言对每个 hybrid 模型开放 `--enable-prefix-caching --no-enable-chunked-prefill`（仅 Qwen3.5-9B 验证过，其他 hybrid 模型未跑）；依赖 stock split 当前 stop 点与私有方法 `_get_local_prefix_cache_hit`/`_resolve_gdn_prefill_backend`/`_mamba_block_aligned_split`，upstream 改名会破坏；causal 请求在此服务器等待而非 chunk；`--max-num-scheduled-tokens` 低于 max-model-len 拒绝，32,514-token 读在 256 decodes 后排队 13.35 s（32,768 预算）vs 1.07 s（33,792）；显式超大 `--max-model-len`（263,168）启动 illegal memory access（stock 亦有）；TP>1 与 GPU preemption 未跑；decision models 不经 Rust 前端服务 `/v1/systemone`。

## 关联

- #63033（open）：PaliGemma 整 prompt 单双向块；#66722（open）：Decision 3.0 prefix_bidir 经 DefaultModelState；#65222（open）：SM 8.x Triton backend 内 FA2 PrefixLM prefill 加速；#64267（RFC）：在 PrefixLMScheduler 上做 D3 state blocks prefix caching。均不与本 PR 重复。
- #46558：FlashInfer 原生 mm-prefix 路径（仅 `is_mm_prefix_lm` 模型，本 PR 未尝试）。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
