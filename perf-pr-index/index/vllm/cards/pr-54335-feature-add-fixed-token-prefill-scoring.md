# PR #54335: [Feature] Add fixed-token prefill scoring

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aoshen02 |
| 合并 | 2026-09-28 |
| 分类 | Other(tier: major) |
| Label | documentation, performance, frontend, ready, mrv2, rust, scheduler |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「15 file(s) dropped from diff for size」 |
| 链接 | https://github.com/vllm-project/vllm/pull/54335 |

## 一句话结论

为 OPD/top-k 蒸馏新增按请求的固定 token prefill logprob 打分，约比全行 `prompt_logprobs` 便宜 10x。

## 动机

OPD / top-k 蒸馏需要在输出后缀对应的 prompt 行上，取得学生模型对一组固定 teacher 候选 token ID 的概率。现有 `prompt_logprobs` 会返回全词表或为每个 prompt 行打分并带一个前导 `None` sentinel，且返回 `list[dict[int, float]]` 对 1000 行、242 ID 的请求要约 22 MB Python 对象，而数组仅约 1 MB；top-k 蒸馏也希望直接拿到数组而非再展平的 dict。因此新增请求级、跨位置固定的候选 ID 打分。

## 优化点

(1) 新增 `SamplingParams.prompt_logprob_token_ids`（候选 ID）与 `prompt_logprob_start`（绝对首个因果 prompt 行）；因 hidden state 行 i 预测 token i+1，OPD 请求设 `start=C-1` 即可只取输出后缀所需行，结果不含前导 `None`。(2) 对每个 prefill chunk，runner 把绝对 start 转成 chunk 内相对行区间，只在该区间跑 lm-head 并 gather 请求的 ID；worker 每请求维护一个 `[num_scored_rows, num_ids]` float32 buffer，首个 chunk 分配、逐 chunk 原地写、末个 chunk 发出，中间内存由后缀长度 × 候选宽度决定。(3) 打分复用 sampler 的 `compute_token_logprobs()` kernel（fp32 累积 max/logsumexp、不物化 `[tokens, vocab]`）；该 kernel 新增 `token_ids` 的 row stride，使请求级 ID 以 stride 0 的广播 `[rows, num_ids]` 视图传入，数值上与 `prompt_logprobs` 逐位一致（在 logits 自身 dtype 内归约会差到 7.78e-02）。(4) 候选数受既有 `max_logprobs` 约束，仅 V2 GPU runner 实现，`/inference/v1/generate` 以 base64 `.npy` float32 返回，并对抢占/部分打分 prefill 加守卫。

## 关键改动

```text
vllm/sampling_params.py: 新增两个参数、skip_reading_prefix_cache 联动及 _validate_logprobs 校验
vllm/v1/worker/gpu/model_runner.py: chunk 内相对行区间跑 lm-head、逐 chunk 写 buffer
vllm/v1/engine/logprobs.py: LogprobsProcessor 持有结果、update_from_output 与 DELTA pop
vllm/v1/core/sched/scheduler.py: boundary 处 trim_reasoning_for_advance + accept_tokens 推进后缀
vllm/outputs.py 与 rust/src/engine-core-client/*: RequestOutput 字段、.npy 编解码与 Rust frontend 转发
```

关键片段（新增参数）：

```diff
+    prompt_logprob_token_ids: list[int] | None = None
+    """Token IDs to score at each scored causal prompt row, where row i scores
+    them as predictions of prompt token i + 1. The last prompt row is excluded,
+    so the result has `max(prompt_len - 1 - prompt_logprob_start, 0)` rows."""
+    prompt_logprob_start: int | None = None
+    """First causal prompt row to score; defaults to the first row."""
```

## 性能声明

环境为 vLLM nightly `0.28.1rc1.dev388+g8a728663c`、4x GB200；模型 `Qwen3-30B-A3B-Thinking-2507`、TP4、8192-token context、median of 5 reps：baseline（无打分）75.2 ms；1000×64 → 77.1 ms（+2.0 ms）、1000×242 → 77.7 ms（+2.6 ms）、1000×1024 → 80.6 ms（+5.5 ms）、4000×64 → 80.7 ms（+5.6 ms）、4000×242 → 81.7 ms（+6.6 ms）、4000×1024 → 88.9 ms（+13.7 ms）；`prompt_logprobs=0` 全 8192 行 → 146.2 ms（+71.0 ms）。正文称在目标蒸馏区间约比全行路径便宜 10x。另一组显示宽候选会在约 40M `row×id` 处交叉（4000×16384 → 176.0 ms，+99.4 ms）。功能关闭时的开销 A/B：单 GB200、Qwen3-0.6B、256 请求×256 greedy decode steps，4.059/4.121 ms/step（base）vs 4.138/4.094 ms/step（branch），轮次漂移 1.5% 大于 A/B 差异。

## 验证与回归风险

正确性（Qwen3-0.6B、193-token prompt、chunk 64）：chunked vs unchunked 行数一致、max abs diff `0.00e+00`，vs `prompt_logprobs=-1` 亦 `0.00e+00`，结果 `(92, 7)` float32，row offset 扫描 offset 0 比相邻偏移拟合好 47x。测试：`test_prompt_logprob_token_ids_with_chunking_and_preemption`（去掉守卫会以 `assert not prompt_token_id_logprobs` 失败）、`test_logprobs_processor.py` 3 passed、`test_logprobs.py -k logprob_token_ids` 6 passed、`cargo test -p vllm-engine-core-client` 102/113 passed、Rust frontend e2e 17/17。风险：仅 V2 GPU runner（否则 `VLLMValidationError` 要求 `VLLM_USE_V2_MODEL_RUNNER=1`）；Rust client 位置解码定长，旧 `vllm-rs` 会报 `array had incorrect length, expected 17`；依赖 #57269 处理 KV connector prefix hit；preemption/流式输入的部分打分需守卫。

## 关联

RFC #56860（本 PR 为 M1，M2 每位置候选列表）；#43463（main 上 `logprob_token_ids`）；#39351、#49171（互补的 open PR）；#57269（KV connector prefix hit 修复，本 PR 依赖）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
