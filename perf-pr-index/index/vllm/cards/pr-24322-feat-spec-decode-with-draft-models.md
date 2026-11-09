# PR #24322: feat: spec decode with draft models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tomasruizt |
| 合并 | 2026-01-19 |
| 分类 | Speculative Decoding \| tier: major |
| Label | documentation, performance, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, qwen, deepseek, gpt-oss, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 6 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/24322 |

## 一句话结论

新增 draft_model 投机解码，Qwen3-32B + Qwen3-1.7B 在 RTX PRO 6000 上 BS1 的 TPOT 缩短约 50%。

## 动机

PR 正文 Purpose：为 speculative decoding (SD) 启用 draft models，例如以 `Qwen3-1.7B` 作 draft、`Qwen3-32B` 作 target；这类 SD 不需要特殊训练头（如 EAGLE 或 Medusa）。示例用法 `--speculative-config '{"model": "Qwen/Qwen3-0.6B", "method": "draft_model", "num_speculative_tokens": 3, "max-model-len": 2000, "disable_padded_drafter_batch": true}'`。Status 称 greedy decoding 下 Qwen3 的接受率很好，且用 SD 的 TPOT 优于不用 SD；正文说明在确保 draft 模型也使用 CUDA graph 之后，SD 吞吐高于不用 SD。

## 优化点

引入 `method="draft_model"` 的投机解码路径：用独立 draft 模型生成候选、target 模型做验证，无需 EAGLE/Medusa 训练头。diff 层面新增 `vllm/v1/spec_decode/draft_model.py` 作为核心实现；`vllm/config/speculative.py`、`vllm/config/vllm.py`、`vllm/engine/arg_utils.py` 接入配置与校验；`vllm/config/parallel.py`、`vllm/v1/attention/backend.py` 与 `backends/utils.py`、`vllm/v1/core/sched/scheduler.py`、`model_loader/*` 做配套；`vllm/v1/spec_decode/eagle.py` 同步调整，benchmarks（datasets.py、ready_checker.py）与 tests 亦有改动。正文未给出更细的 kernel/调度机制描述。

## 关键改动

```text
vllm/v1/spec_decode/draft_model.py: 新增 draft model 投机解码实现
vllm/config/speculative.py: 新增 draft_model method 及相关配置
vllm/config/vllm.py、vllm/engine/arg_utils.py: 配置接入与校验
vllm/v1/core/sched/scheduler.py: 调度器配套改动
vllm/v1/spec_decode/eagle.py: EAGLE 路径适配
```

## 性能声明

Acceptance Length（命令 `VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py --model-dir Qwen/Qwen3-32B --draft-model Qwen/Qwen3-1.7B --method draft_model --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num_prompts 100 --temp 1.0 --gpu-memory-utilization 0.9`）：t=0.0 时 Qwen3-32B/Qwen3-1.7B K=3 AL 2.79、K=4 AL 3.12，Llama-3.2-1B 自身 K=3 AL 3.97；t=1.0 时分别 2.61、2.85、2.82。Online Throughput（RTX PRO 6000 96GB，`vllm serve Qwen/Qwen3-32B --max-model-len 20000`，`vllm bench serve --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --max-concurrency (100|1) --temperature 0.0`）：Batch Size 1 / t=0.0 无 SD 943s / TTFT 69.14 / TPOT 46.06 / ITL 45.88，有 SD 466s / 76.77 / 22.62 / 62.44（正文 "runtimes and TPOT are shorter by ~50%"）；Batch Size 100 / t=0.0 16.88s / 262.59 / 65.09 / 64.84 → 13.04s / 284.45 / 43.70 / 121.48，t=1.0 16.83s / 230.04 / 64.95 / 64.70 → 14.84s / 272.51 / 48.00 / 122.27。正文："Using SD the TPOT is 50% shorter (better) in batch size 1, and 26% to 33% shorter in batch size 100"；"Using SD the TTFT and ITL are higher (worse), because tokens are produced in batches by the spec-decoding"。

## 验证与回归风险

Test Plan：`cd tests/v1/e2e/; pytest test_spec_decode.py -k test_draft_model_correctness`，正文说明新增单测检查 correctness metrics；EAGLE testing 用 `--method eagle3` 验证 "EAGLE implementation stays unaffected"，结果与 #17504 的历史测量一致。回归风险：正文指出 t=1.0 下 AL 下降，因 "spec-decode with probabilities is not yet implemented, needed for lossless rejection sampling"，需等 #20459 合并后改善；SD 的 TTFT 与 ITL 变差；默认模型 `meta-llama/Llama-3.1-8B-Instruct` 未跑（未找到合适 draft 模型）；Profiling 章节因 draft 模型改用 CUDA graph 后已移除。正文另给出 EAGLE 不受影响的验证命令与 total_num_output_tokens: 16990 的结果，并说明所有脚本与日志存放于外部 Google Drive 链接。

## 关联

正文明确给出 #20459（概率 spec-decode / lossless rejection sampling，待其改善非 greedy 的 AL）、#24575（正文称 `--profile` flag 由该 PR 引入）、#17504（EAGLE 历史测量对照）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
