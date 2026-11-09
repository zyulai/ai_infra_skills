# PR #12094: [V1][Perf] Reduce scheduling overhead in model runner after cuda sync

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @youngkent |
| 合并 | 2025-01-26 |
| 分类 | Overlap / Scheduler（tier: major） |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/12094 |

## 一句话结论

把 decode 迭代后 CPU bookkeeping 挪到 cuda sync 前，scheduling overhead 约降 20%、e2e latency 降 1-2%。

## 动机

decode 迭代之后 model runner 会做一些 CPU bookkeeping 操作；PR 认为可以在等待 cuda sync 期间并行完成这些 bookkeeping，cuda sync 之后只做简单快速的更新，从而降低 decode 迭代之间的 scheduling overhead（PR 标注约 20%，依据所附 GPU trace）。

## 优化点

把 `SamplerOutput.sampled_token_ids` 从 `List[int]` 改为 `torch.Tensor`；`sampler.py` 不再调用 `sampled.tolist()` 触发 CPU-GPU 同步，而是保留 GPU tensor。`gpu_model_runner.py` 的 `execute_model` 把 req_ids 断言 / 构造与状态更新前移到 GPU→CPU sync 点之前（先用 `append(0)` 占位并记录 `request_seq_lens`），在 sync 点一次性 `sampler_output.sampled_token_ids.tolist()`，之后仅做简单快速的回填（`token_ids_cpu[i, seq_len] = token_id`、`output_token_ids[-1] = token_id`）。这样把尽可能多的 CPU 操作放到同步点之前，sync 之后只保留必要更新。

## 关键改动

```text
vllm/v1/outputs.py: SamplerOutput.sampled_token_ids 由 List[int] 改为 torch.Tensor
vllm/v1/sample/sampler.py: 移除 sampled.tolist()，sampled_token_ids 直接保留 GPU tensor，延迟 CPU-GPU 同步
vllm/v1/worker/gpu_model_runner.py: execute_model 把 bookkeeping 前移到 sync 之前（append(0) 占位 + request_seq_lens），sync 后一次性 tolist() 并回填真实 token id
```

## 性能声明

测试条件：`VLLM_USE_V1=1 python3 benchmarks/benchmark_latency.py --model "/data/users/ktong/llama/llm_8b_oss" --tensor-parallel-size 1 --input_len 1000 --batch_size 32`（8B 模型、TP1、input_len 1000、batch_size 32）。PR 标注 "The change should reduce scheduling overhead between decode iterations by ~20%" 与 "1-2% e2e latency reduction"。给出的 Output：Avg latency 2.338167402730323 s、10% percentile 2.3207896508742123 s、25% 2.3264574960339814 s、50% 2.3333765944698825 s、75% 2.343035737867467 s、90% 2.3567665563430635 s、99% 2.3934816433605737 s。正文仅给出这一组延迟数值，未附优化前对照数值，也未注明硬件型号。

## 验证与回归风险

PR 正文未列出单测；验证依据为所附 GPU trace（before / after 截图）与上述 `benchmark_latency.py` 的 e2e latency benchmark。回归风险：改动覆盖 V1 `SamplerOutput` 的 `sampled_token_ids` 类型（`List[int]` → `torch.Tensor`）、sampler 的输出构造，以及 `gpu_model_runner.execute_model` 的 bookkeeping 顺序与 `token_ids_cpu` / `output_token_ids` 的回填路径；正文未给出其他受影响模型、硬件、精度或已知回归信息。

## 关联

无（PR 正文未给出其他 PR 编号线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
