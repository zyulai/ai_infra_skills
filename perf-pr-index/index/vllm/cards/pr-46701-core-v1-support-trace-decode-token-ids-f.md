# PR #46701: [Core][V1] Support trace_decode_token_ids for deterministic decode replay

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zllion |
| 合并 | 2026-08-20 |
| 分类 | Speculative Decoding \| tier: major |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, mrv2, mistral, rust, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)，pack 末尾标注 2 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/46701 |

## 一句话结论

新增 `trace_decode_token_ids` 强制 v1 sampler 逐步重放固定 decode 轨迹以精确比对 logprobs，PR 未给出加速数字。

## 动机

PR Purpose：让请求沿固定 token trace 并返回模型赋予该 trace 的 logprobs。用途有二：量化不同 settings/backends（dtype、quantization、attention/MoE backend、TP size 等）在相同 token 序列上的数值差异；RL 训练中在训练引擎下重算 rollout 精确 token 的 logprobs，以度量并缩小 rollout↔train logprob gap，避免重新采样漂移到不同轨迹。为何是 decode-replay 而非 prefill-only scoring：prefill 整段 (prompt+response) 不能忠实复现 decode 时分布，因为许多模型在 prefill 与 decode 路径使用不同算子/内核（attention、MoE kernel 不同），要测量 decode 实际产生的 logprobs 必须让 token 逐步走 decode 路径。

## 优化点

新增 `SamplingParams.trace_decode_token_ids` 字段，强制 v1 sampler 逐步发出预定 decode token IDs 而非采样；真实 logprobs 与 token ranks 仍从未修改的 logit 分布计算。sampler 在采样之后注入每个 trace token（`Sampler._inject_trace_tokens` / `TraceReplayState.apply_trace`），使 logprobs 保持真实（注入前基于原始 logits 计算）。`InputBatch` 与 `spec_token_ids` 并行跟踪每请求 trace，per-request step index 计真实输出 token、跳过尾部 `-1` async-scheduling 占位符。`EngineCore.add_request` 把生成上限截到 trace 长度并强制 `ignore_eos`，使 trace 内 EOS 不提前终止；对不兼容模式（speculative decoding、`n > 1`）回退普通采样并告警。新增 `TraceReplayState`、配置开关、文档 `docs/serving/online_serving/trace_replay.md` 与 demo `examples/generate/trace_replay_offline.py`。

## 关键改动

```text
vllm/sampling_params.py: 新增 trace_decode_token_ids 字段与校验（空/负/非 int 拒绝）。
vllm/v1/worker/gpu/sample/sampler.py: 引入 TraceReplayState，采样后调用 apply_trace 覆盖 sampled tokens。
vllm/v1/worker/gpu/sample/trace_replay.py: 新增 trace 注入状态与逐请求 step 跟踪。
vllm/v1/engine/input_processor.py、vllm/config/*、engine/arg_utils.py: trace 长度截断、ignore_eos 与开关。

  if self.trace_replay_state is not None:
      # Overwrite sampled tokens with the replay trace up-front so that
      # computed logprobs reflect the real distribution of the forced token.
      self.trace_replay_state.apply_trace(sampled, idx_mapping)
```

## 性能声明

PR 正文未给出定量性能数据（regex 未检测到 perf claims），也未提供 benchmark 与测试条件；该 PR 为确定性重放/诊断能力，正文明确其目的是比对 logprobs 数值差异而非加速。diff 中 `vllm/config/model.py` 附近的既有 batch-invariance 配置注释提及「significantly lower throughput on most GPUs」，属既有说明、非本 PR benchmark。

## 验证与回归风险

单测（CPU-only）：`.venv/bin/python -m pytest tests/v1/sample/test_trace_replay.py tests/v1/engine/test_engine_core_trace_replay.py -v` → 17 passed, 16 warnings in 0.14s。覆盖：SamplingParams 字段默认值/clone/校验（空/负/非 int 拒绝）；sampler injection 含 step 0、混合 per-request steps、past-end-of-trace(no-op)、async `-1` placeholder、unset/none；engine-core admission（生成上限截到 trace 长度 + 强制 `ignore_eos`、speculative decoding 与 `n>1` 回退）。`pre-commit run --all-files` 与 `mypy-3.12`（CI hook stage）在改动文件通过（ruff-check、ruff-format、SPDX headers、forbidden-imports）。Demo `examples/generate/trace_replay_offline.py` 先 greedy 捕获 decode tokens 再重放并逐 token 断言匹配、打印 per-token logprobs；PR 明确警告单测在 CPU-only 机器运行，端到端 demo 应在 GPU 上确认逐 token 重放。风险与边界：不兼容模式（speculative decoding、`n>1`）回退普通采样并告警；因注入在采样后，logprobs 保持真实；该功能强制改变输出 token 序列，仅用于诊断/RL。

## 关联

无（PR 正文「Not a duplicate」段说明搜索 open PR 无匹配结果，仅提及既有 `logprob_token_ids`、`spec_token_ids` 功能，未引用具体 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
