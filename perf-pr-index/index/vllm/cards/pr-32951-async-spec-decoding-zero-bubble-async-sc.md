# PR #32951: [Async][Spec Decoding] Zero-bubble async scheduling + spec decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-03-23 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明「3 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/32951 |

## 一句话结论

spec decoding 乐观假设 draft 全接受并延后修正，H100 DeepSeek-V3.2 MTP 下 mean TPOT 9.19→8.90 ms（3%）。

## 动机

本 PR 是 #29957 的重构，旨在提升 spec decoding 的异步性：在 CPU 上乐观假设所有 draft tokens 都被接受，把修正延后到 forward pass 之后，并以 GPU 侧 tensors 为 source of truth。model runner V2 的 `compute_slot_mappings_kernel`（含 @yewentao256 在 #34179 的改动）被适配进 V1。

## 优化点

CPU 侧不再等待 draft 接受结果，而是乐观假设所有 draft tokens 被接受、把修正延后到 forward pass 之后，GPU 侧 tensors 作为 source of truth；`compute_slot_mappings_kernel`（model runner V2 + #34179 改动）适配进 V1。diff 显示：`vllm/config/vllm.py` 在 `speculative_config` 非空、`async_scheduling` 开启且未 `disable_cascade_attn` 时打 warning 并强制置 `disable_cascade_attn=True`（cascade attention 与 async speculative decoding 尚不兼容）；`eagle.py` 移除 `self.runner` 依赖、`prepare_next_token_ids_padded` 的签名由 `common_attn_metadata` 改为 `seq_lens_cpu`；并改动 `block_table.py`、`gpu_input_batch.py`、`gpu_model_runner.py`、`spec_decode/utils.py`、`extract_hidden_states.py` 等配合 slot mapping 与 input batch 的异步路径。

## 关键改动

```text
vllm/v1/spec_decode/eagle.py: prepare_next_token_ids_padded 改以 seq_lens_cpu 为输入、移除 runner 依赖，配合乐观接受假设
vllm/config/vllm.py: async_scheduling + spec decode 时强制 disable_cascade_attn=True（cascade attention 暂不兼容）
vllm/v1/worker/gpu_model_runner.py: 把 compute_slot_mappings_kernel（V2 + #34179）适配进 V1
vllm/v1/worker/block_table.py、vllm/v1/worker/gpu_input_batch.py: 配合 slot mapping 与 input batch 的异步改动
vllm/v1/spec_decode/utils.py、vllm/v1/spec_decode/extract_hidden_states.py: spec decode 辅助路径调整
```

## 性能声明

测试条件：H100，`vllm serve deepseek-ai/DeepSeek-V3.2 -tp 8 -ep --no-enable-prefix-caching --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`，配合 `vllm bench serve --dataset-name spec_bench --dataset-path question.jsonl --spec-bench-output-len 2048 --num-prompts 10 --max-concurrency 1 --seed 42 --ignore-eos --temperature 0 --tokenizer deepseek-ai/DeepSeek-V3`。PR 正文结论：TPOT 9.19 ms -> 8.90 ms（3% speedup）。Main：Mean TTFT 62.86 ms、Mean TPOT 9.19 ms（Median 9.19 / P99 9.51）、Mean ITL 17.79 ms、Output token throughput 108.46 tok/s、Total 112.70 tok/s、Acceptance rate 93.56%、Acceptance length 1.94、Drafts 10578、Accepted 9897。PR：Mean TTFT 53.30 ms、Mean TPOT 8.90 ms（Median 8.89 / P99 9.20）、Mean ITL 17.27 ms、Output 112.04 tok/s、Total 116.42 tok/s、Acceptance rate 94.05%、Acceptance length 1.94、Drafts 10551、Accepted 9923。

## 验证与回归风险

PR 正文未列出专门单测；Test Plan 为 H100 上的 e2e `vllm serve` + `vllm bench serve`（10 prompts、max-concurrency 1、temperature 0）。回归风险：cascade attention 在 async speculative decoding 下被强制禁用（`disable_cascade_attn=True`，PR 注明 "not yet compatible with async speculative decoding"）；改动覆盖 spec decode 的 slot mapping、input batch、`eagle.py` proposer 与 config 校验路径；正文未给出其他受影响模型、硬件、精度或已知回归信息。

## 关联

#29957（本 PR 是其重构）；#34179（@yewentao256 对 `compute_slot_mappings_kernel` 的改动，本 PR 适配进 V1）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
