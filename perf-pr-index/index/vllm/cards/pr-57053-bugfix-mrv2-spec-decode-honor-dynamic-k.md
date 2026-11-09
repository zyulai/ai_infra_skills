# PR #57053: [Bugfix][MRV2][Spec Decode] Honor dynamic K in autoregressive speculators, 1.2~1.3x kernel perf improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-10-05 |
| 分类 | Attention / Kernel（tier: major） |
| Label | bug, speculative-decoding, ready, nvidia, mrv2, dflash |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/57053 |

## 一句话结论

MRV2 向 autoregressive speculator 传入动态 K，合成内核基准提速 1.28~1.32x。

## 动机

MRV2 没有把 `SchedulerOutput.num_spec_tokens_to_schedule` 传播给 autoregressive speculators，`propose()` 始终用最大 `K`（`num_speculative_steps`，脚本中 MAX_K=5）而非 scheduler 选出的运行时 K（脚本中 3），导致不必要的 draft forwards 与 stale draft tokens。本 PR 修复该传播。

## 优化点

`GPUModelRunner.execute_model` 把 `scheduler_output.num_spec_tokens_to_schedule` 存入 `ExecuteModelState`，`sample_tokens` 取出后作为 `num_speculative_tokens` 传给 `speculator.propose(...)`。`AutoRegressiveSpeculator.propose` 新增可选参数 `num_speculative_tokens`（缺省回退 `self.num_speculative_steps`），`draft_max_seq_len` 与 early-exit 判定改用它，K<=1 时把 `draft_tokens` 填 -1 并早退。model_runner 对 `num_spec_tokens < self.num_speculative_steps` 的尾部置 -1，并把 `draft_tokens` 按 `[:num_spec_tokens]` 切片交给 handler，避免 stale draft tokens。autoregressive 的 CUDA graph capture 在启用 dynamic spec decode 时以 `specialize_spec_tokens` 特化，使不同 K 有对应 graph；dflash / draft_model / multi_module_mtp 等其他 speculator 与 cudagraph、warmup 路径同步适配签名。

## 关键改动

```text
vllm/v1/worker/gpu/model_runner.py: ExecuteModelState 增加 num_spec_tokens_to_schedule；sample_tokens 取出并作为 num_speculative_tokens 传给 propose，尾部置 -1、按 K 切片 draft_tokens。
vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py: propose 新增 num_speculative_tokens 参数，用于 draft_max_seq_len 与 early-exit；capture 用 specialize_spec_tokens。
vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py、vllm/v1/worker/gpu/cudagraph_utils.py: 支持按 spec tokens 特化的 graph capture。
vllm/v1/worker/gpu/spec_decode/{dflash,draft_model,multi_module_mtp}/speculator.py、spec_decode/speculator.py: 同步 propose 签名。
tests/v1/worker/test_gpu_autoregressive_speculator.py、test_gpu_model_runner_v2.py、tests/v1/cudagraph/test_cudagraph_manager.py: 相关单测更新。
```

```text
                     self.sampler.sampling_states.seeds.gpu,
                     num_speculative_tokens=num_spec_tokens,
                     dp_sync=dp_sync,
                     mm_inputs=mm_inputs,
                 )
             if num_spec_tokens < self.num_speculative_steps:
                 draft_tokens[:, num_spec_tokens:] = -1
```

## 性能声明

标题声明 "1.2~1.3x kernel perf improvement"。正文给出 AI generated script（合成基准）输出：`SyntheticSpeculator`（把 model forward 换成 matmul）、`MAX_K=5`、`MAX_BATCH_SIZE=48`、`HIDDEN_SIZE=4096`、`DYNAMIC_K_SCHEDULE=[(1, 48, 3)]`、CUDA、`triton.testing.do_bench`。结果（batch / sched_k / buggy_ms / fixed_ms / speedup / forwards buggy→fixed）：1/3/0.151/0.115/**1.306x**/5→3；2/3/0.148/0.115/**1.286x**/5→3；4/3/0.149/0.115/**1.298x**/5→3；8/3/0.148/0.115/**1.282x**/5→3；16/3/0.151/0.115/**1.315x**/5→3；32/3/0.150/0.114/**1.306x**/5→3；48/3/0.149/0.115/**1.287x**/5→3。buggy 版 `propose` 省略运行时 K 用 max K=5，fixed 版传 scheduler 选出的 K=3，forwards 由 5 降到 3。该表为合成 matmul 内核计时，PR 未注明测试条件（无真实模型 / 硬件型号 / 精度 / workload，非端到端 serving）。

## 验证与回归风险

正文称 "Acc covered in current unit test"，perf 由上述 AI generated script 给出；diff 更新了 `tests/v1/worker/test_gpu_autoregressive_speculator.py`、`test_gpu_model_runner_v2.py`、`test_gpu_model_runner_v2_eplb.py`、`test_gpu_warmup_blocks.py`、`tests/v1/cudagraph/test_cudagraph_manager.py` 等单测。风险：改动触及 MRV2 propose 调用链、dynamic spec decode、CUDA graph capture（按 spec tokens 特化）与 draft token 传递，可能影响 dflash / draft_model / multi_module_mtp 等其他 speculator、EPLB 与 warmup 路径；`num_speculative_tokens` 缺省回退 `num_speculative_steps` 以保持非动态路径行为。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
