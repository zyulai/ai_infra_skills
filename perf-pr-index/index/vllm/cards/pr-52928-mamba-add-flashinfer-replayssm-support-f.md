# PR #52928: [Mamba] Add FlashInfer ReplaySSM support for MTP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @askliar |
| 合并 | 2026-09-30 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, ci/build, mrv2 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「9 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/52928 |

## 一句话结论

ReplaySSM 扩展到 MTP，GB200 batch256 达 28,703 tok/s，较 Triton 快 5.65%。

## 动机

正文说明这是 stacked follow-up，把 #52506 中原生 FlashInfer ReplaySSM 集成从单 token decode 扩展到 Mamba2 模型的 multi-token prediction（MTP）验证。MTP 下模型要在采样决定接受多少之前，评估一个 target token 加若干 draft token；ReplaySSM 的做法是为避免给每个投机 token 物化一份独立的 recurrent state，让每层保留一个 checkpoint 与一个输入 ring，接受长度在下次 forward 前推进共享 cursor，只有当下一个验证窗口会溢出 ring 时才更新 checkpoint。正文明确本 PR 堆叠在 #52506 之上、不可独立合并，FlashInfer 侧还依赖 flashinfer-ai/flashinfer#4545 的兼容 ReplaySSM API 与原生 autotuning。

## 优化点

MTP-shaped execution：均匀验证批次用 dense `(batch, tokens, ...)` 输入以便 FlashInfer 选择原生 MTP tactics，变长批次用打包的 `cu_seqlens` 接口。每个 cache group 只做一次 acceptance update——所有 Mamba 层观察到相同接受 token 数，ring cursor 共享、每次 model forward 推进一次，checkpoint 与 ring payload 仍层内局部。每个请求只存一个物理 Mamba state page（而非每个 target/draft 位置一个），causal-conv1d 验证长度与物理 block-table 宽度分开传递。CUDA-graph-safe indexing：连续的 FlashInfer state-index 向量放在 runner 持久存储中、replay 前刷新以保持捕获地址稳定。ReplaySSM 调用使用 FlashInfer 的 `checkpointing_ssu` API 与原生 autotuning，PDL 显式透传但默认关闭（正文称并非一致的端到端收益）。本 PR 不重复 #49847 的 Triton 实现（后者用独立 Triton kernel、token-major ring 与自有 cursor 生命周期），而是保留 FlashInfer 布局并把 update 委托给原生 FlashInfer；不修改 FlashInfer、不加 prefix-cache 对齐，ReplaySSM 继续用 `mamba_cache_mode=none`。

## 关键改动

```text
vllm/model_executor/layers/mamba/mamba_mixer2.py: 新增 _view_mtp_decode_tensor、_replayssm_prev_query_len/_commits_replayssm_trackers，接入 MTP 形态的 ring tracker 提交
vllm/model_executor/layers/mamba/abstract.py: Mamba 层抽象适配（diff 中改动）
vllm/config/cache.py、vllm/config/vllm.py: 配置侧适配
.buildkite/test_areas/engine.yaml: 新增 (H100) ReplaySSM E2E 步骤，跑 v1/e2e/test_replayssm_decode.py
tests/kernels/mamba/test_ssu_dispatch.py、tests/model_executor/test_replayssm_warmup.py、tests/v1/attention/test_replayssm_metadata_builder.py: 相关单测
```

```python
def _view_mtp_decode_tensor(
    tensor: torch.Tensor, decode_batch: int, spec_query_len: int
) -> torch.Tensor:
    return tensor.view(decode_batch, spec_query_len, *tensor.shape[1:])
```

## 性能声明

测试条件（正文 Actual-model throughput）：模型 `NVIDIA/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`，MTP 三个 draft token 与四 token 验证窗口，实测 acceptance length 4.0，ReplaySSM buffer 16，SSM state FP32，stochastic rounding 关闭，CUDA graphs 开启，指标为 output tokens/s。

H100 80GB：

| Batch | FlashInfer PDL off | FlashInfer PDL on | Triton #49847 | Best FI vs Triton |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 6,578.81 | 6,563.84 | 6,472.75 | +1.64% |
| 128 | 19,343.22 | 19,388.90 | 19,059.60 | +1.73% |
| 256 | 21,896.57 | 22,014.13 | 21,725.24 | +1.33% |
| 512 | 24,863.34 | 24,852.45 | 24,458.58 | +1.65% |

GB200：

| Batch | FlashInfer PDL off | FlashInfer PDL on | Triton #49847 | Production FI off vs Triton |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 8,811.71 | 7,921.11 | 8,467.73 | +4.06% |
| 128 | 25,532.13 | 25,948.60 | 24,270.93 | +5.20% |
| 256 | 28,687.11 | 28,703.49 | 27,153.46 | +5.65% |
| 512 | 29,272.09 | 29,170.54 | 27,718.29 | +5.61% |

正文指出 PDL 在 GB200 B16 有害、大 batch 基本中性，故生产路径保持关闭。正确性：GB200 上 GSM8K matched eval，FlashInfer 437/512（85.35%）、0 invalid，Triton #49847 437/512（85.35%）、0 invalid；FP32 SSM state、stochastic rounding 关闭；严格 H100 B16/AL1 CUDA-graph 对照两 backend 完整输出 hash 一致。Nsight Systems profile（64 个稳态 CUDA-graph decode step、累加全部 23 个 Mamba 层）：H100 B16/AL4 FlashInfer kernels 739.42 μs/step、Triton 644.74 μs/step，tracker 6.89 对 10.87 μs/step；B128/AL4 为 3,529.11 对 3,587.91 μs/step、tracker 8.50 对 12.69 μs/step。GB200 B256 下 ReplaySSM 约占 execute-step GPU 时间 16%，1.56× in-model 改进预测最大整步增益约 6.1%，接近实测 5.65%。缓存容量 B256：FlashInfer 303,923 对 Triton 308,121 tokens（-1.4%）；B512：293,990 对 299,008（-1.7%）。

## 验证与回归风险

Validation：changed-file pre-commit 通过（Ruff、mypy、SPDX、configuration validation、forbidden imports、CUDA API checks）；focused H100 suite 41 passed、200 deselected；GB200 GSM8K 精度完全一致、零 invalid outputs；TP2 actual-model MTP smoke 完成。可能影响的路径：Mamba2 模型在 FlashInfer 后端下启用 MTP 的 decode/verify、ReplaySSM ring checkpoint 与 state page 分配、CUDA-graph replay 的 state-index 地址稳定性；正文注明不改 FlashInfer、不做 prefix-cache 对齐、继续 `mamba_cache_mode=none`，PDL 默认关闭，且依赖 #52506 与 flashinfer-ai/flashinfer#4545。pack 末尾注明 9 file(s) dropped from diff。

## 关联

正文明确引用：#52506（单 token decode 的原生 FlashInfer ReplaySSM 集成，本 PR 堆叠其上、不可独立合并）、#49847（用独立 Triton kernel 实现 MTP ReplaySSM 的对照方案）、flashinfer-ai/flashinfer#4545（兼容 ReplaySSM API 与原生 autotuning，外部依赖）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
