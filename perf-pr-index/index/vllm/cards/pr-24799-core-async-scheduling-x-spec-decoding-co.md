# PR #24799: [Core] Async Scheduling X Spec Decoding Compatibility

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Ronald1995 |
| 合并 | 2025-11-17 |
| 分类 | Speculative Decoding \| tier: major |
| Label | speculative-decoding, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/24799 |

## 一句话结论

让 async_scheduling 与 eagle spec decode 兼容，A100 上 96 prompts 吞吐较 sync +7.1%。

## 动机

正文说明 #19970 实现 async_scheduling、#23569 在其上实现 `prepare_input` overlap、#24539 重构 eagle spec_code 使其不再依赖 CPU 的 `sample_token_ids`。本 PR 基于 #24539，目标是让 spec decode 与 async_scheduling 同时启用：两者一起时不再把 draft token ids 复制回 scheduler，而是缓存在 `gpu_model_runner` 中，并用 `_draft_token_ids` 直接更新下一步 `execute_model` 的 `input_ids`。因 ngram 与 medusa 目前仍依赖 CPU 的 `sample_token_ids`，本 PR 只支持 eagle spec_decode 与 async_scheduling 的组合，其余留待后续重构。

## 优化点

在同时启用 async_scheduling 与 spec decode 时，取消把 draft token ids 回传 scheduler，改为在 `gpu_model_runner` 内缓存 `_draft_token_ids`，并在下一步 `execute_model` 直接用它更新 `input_ids`，去掉 CPU↔scheduler 的 draft token 拷贝。改动横跨 V1 的 scheduler（`async_scheduler.py` / `scheduler.py`）、engine core 与 processor、speculative 与 vllm 配置、`eagle.py`、`gpu_input_batch.py`、`gpu_model_runner.py` 及 logits_processor 注册。e2e 测试取消 `test_with_spec_decoding` 的 skip、把 MTP model 换成 `meta-llama/Llama-3.2-1B-Instruct`、spec method 改为 `eagle3` 并调整 `min_tokens` / `max_tokens`。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: 缓存 _draft_token_ids 并用其直接更新下一步 execute_model 的 input_ids
vllm/v1/spec_decode/eagle.py / vllm/v1/core/sched/async_scheduler.py: 适配 eagle spec decode 与 async scheduling 共存
vllm/config/speculative.py / vllm/config/vllm.py / vllm/v1/engine/core.py / processor.py: 相关配置与调度路径接入
tests/v1/e2e/test_async_scheduling.py: 启用 async_scheduling + spec decoding 的 e2e 一致性测试
```

## 性能声明

Test device: Nvidia A100。测试配置：dataset 为 prm800k（读 jsonl 生成 prompts），`SamplingParams(temperature=0, max_tokens=1024)`，`LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", gpu_memory_utilization=0.9, tensor_parallel_size=1, max_model_len=2048, max_num_seqs=128, max_num_batched_tokens=4096, async_scheduling=True, speculative_config={"model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 2, "method": "eagle"}, seed=1234)`。Test Result（tps）：num_prompts 24 时 async_scheduling 2356 对 sync_scheduling 2314，speedup 1.8%；48 时 3759 对 3539，6.2%；96 时 5110 对 4770，7.1%。pack 的 regex 未检测到定量性能声认；PR 未注明精度（dtype）条件。

## 验证与回归风险

Test Plan 为 e2e test（async_scheduling + EAGLE-LLaMA3-Instruct-8B draft model，确保其正常工作）。precision 部分：对比 async_scheduling 与 sync_scheduling 在 speculative decoding 下的输出，正文称"the outputs are actually the same. so the async_scheduling doesn't make precision problem"。diff 更新 `tests/v1/e2e/test_async_scheduling.py` 并涉及 scheduler、engine core、processor、`gpu_model_runner.py` 等 V1 路径；正文未给出测试通过计数，也未逐一列出其他受影响模型或硬件，diff 中 `tests/v1/e2e/test_async_scheduling.py` 的用例说明其覆盖 preemption、executor、async scheduling、prefill chunking 等多种组合下的一致性与 acceptance rate；正文的 precision 结论是 async 与 sync 两种方式在 speculative decoding 下输出相同，因此不引入精度问题。仅明确 ngram / medusa 组合暂不支持。

## 关联

正文明确列出前序 PR：#19970（async_scheduling）、#23569（基于 #19970 的 `prepare_input` overlap）、#24539（eagle spec_code 重构，本 PR 基于它）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
