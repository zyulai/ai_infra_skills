# PR #52033: [Perf][ROCm] Dual-stream decode with hipgraphs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simondanielsson |
| 合并 | 2026-08-30 |
| 分类 | MoE（tier: major） |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/52033 |

## 一句话结论

ROCm DP 下让 shared expert 与 routed expert 双流重叠，1k/1k 与 8k/1k TPOT 约降 3-4%。

## 动机

重开 #48223（因 Qwen3.5 with DP2EP 的 gsm8k 精度差导致 CI 失败，本 PR 前身在 #52024 被 revert）。Root cause：Qwen3.5 用同一 tensor 同时作为 hidden state 与 shared expert 输入，且整个 MoE 都不创建副本；因其未量化 routed experts 输入（量化时会隐式创建 copy，两个 buffer 不同），多流下产生 race condition。Purpose 为 Fixes #48111：启用 CUDA-like 平台的 dual-stream decode 正确重叠并兼容 hip/cudagraph。

## 优化点

Fix：若未量化 routed experts 输入则禁用 dual-stream，且该检查只在 ROCm 做（保留 Nvidia 行为）——使 Qwen3.5 不产生 race、精度保住但多流关闭，DSv3 仍走多流。`moe_runner.py` 新增 `is_multistream_safe`（`not current_platform.is_rocm() or routed_input_is_quantized()`，后者检查 `moe_quant_config.quant_dtype`）传给 `SharedExperts`。关键机制：shared expert 原先在 routed experts 完成后才在 aux stream 发射，导致顺序无重叠；改为在 dispatch 之前通过 `maybe_forward_async` 于 aux stream 提前发射，`_apply_quant_method` 用 `shared_experts_overlapping` 标志改为只 `wait()`。仅 ROCm 且用 DP 时启用（TP 下有性能回归），与 `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` 互斥，可用 `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` 禁用。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/runner/moe_runner.py: 新增 routed_input_is_quantized / is_multistream_safe 门控；_forward_impl 在 dispatch 前以 maybe_forward_async 提前发射 shared expert；_apply_quant_method 用 shared_experts_overlapping 只做 wait()；删除 _maybe_sync_shared_experts_stream。
vllm/model_executor/layers/fused_moe/runner/shared_experts.py: 构造新增 is_multistream_safe 回调，用于决定是否走多流。
```

```text
            # When unquantized, shared expert inputs alias the hidden states,
            # which can lead to race condition in multi-stream mode. Quantized
            # routed experts copy the input into a fresh buffer first, breaking
            # the alias, so overlap is safe. Only observed on ROCm.
            routed_input_is_quantized = lambda: (
                self.routed_experts.quant_method.moe_quant_config is not None
                and self.routed_experts.quant_method.moe_quant_config.quant_dtype
                is not None
            )
            is_multistream_safe = lambda: (
                not current_platform.is_rocm() or routed_input_is_quantized()
            )
```

## 性能声明

**Gain:** "About -3-4% TPOT on 1k/1k and 8k/1k when using DPA vs default"（正文原话）。条件：8xMI300，DSv3（`deepseek-ai/DeepSeek-V3-0324`），`VLLM_ROCM_USE_AITER=1`，DP8+TP1，对比 nightly `vllm/vllm-openai-rocm:nightly-a16dbd5b...` with fused shared experts，1k/1k 与 8k/1k ISL/OSL，10x prompts / 2x warmups，seed 5678，`--max-concurrency 128 --num-prompts 1280 --num-warmups 256`。1k/1k DP8+TP（P50）`% TPOT improved` 按 conc 1/2/4/8/16/32/64/128/256 = +4.24% / +4.58% / +7.03% / +3.41% / -6.77% / +2.75% / +4.71% / +1.86% / +3.12%；8k/1k DP8+TP 按 conc 1/2/4/8/16/32/128 = +3.59% / +5.47% / +2.82% / +5.54% / -3.07% / +11.44% / -3.71%。1k/1k TP8EP 下 TP 比默认 sequential 更差（正文 "TP is worse than (default) sequential execution"），故仅 DP 启用。精度：Qwen3.5 DP2EP gsm8k flexible-extract 0.8453 ± 0.0100 / strict-match 0.8332 ± 0.0103；DSv3 gsm8k this branch 0.9416 ± 0.0065 / 0.9409 ± 0.006 vs nightly 0.9484 ± 0.0061 / 0.9492 ± 0.0060。

## 验证与回归风险

验证：Get traces for DSv3 与 Qwen3.5（DP）、重跑 gsm8k（`--data-parallel-size 2 --enable-expert-parallel`，`Qwen/Qwen3.5-35B-A3B`）、8xMI300 的 DSv3 DPA+TP bench、1k/1k 与 8k/1k bench serve sweep、GSM8k non-regression、B200 确认 NV 行为不变、cudagraph 下 uniform batches 仍用 full cudagraphs。风险：仅在 ROCm + DP 启用（TP 下性能回归）；与 `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` 互斥；未量化 routed expert 输入时禁用双流（Qwen3.5 精度保住但无重叠）；正文注明 "TTFT numbers seems highty variable"，且 FSE 可用时 TPOT 优于 dual-stream。

## 关联

#52024（本 PR 前身被 revert）、#48223（re-opened 的原 PR，因 Qwen3.5 DP2EP gsm8k 精度失败，其 comment 记录 TP 下性能回归）、#48111（Fixes）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
