# PR #56503: [ROCm][DSV4.1][Perf] Use AITER mHC for the delayed pre block

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Fangzhou-Ai |
| 合并 | 2026-09-12 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, deepseek, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/56503 |

## 一句话结论

把 V4.1 ROCm 的 delayed mHC pre 迁到 AITER，每 seam 141→4 次 launch（35x 更少），CUDA graph A/B 下最快 21.30x。

## 动机

DeepSeek-V4.1-Flash 在 AMD 上仍以 eager Torch reference 跑 mHC block（`vllm/models/deepseek_v4_1/amd/model.py` 直接 import `mhc_pre_delayed_torch` / `mhc_post_torch`）；DeepSeek-V4 已迁到 `MHCPreOp` / `MHCPostOp` dispatch 层并接上 AITER（#41946、#43950、#52737），V4.1 从未迁移，MI355X 上每个 sublayer seam 为 Sinkhorn-normalize 一个 4x4 矩阵付出约 141 次 kernel launch。MI355X（gfx950, TP4）decode trace 的 per-layer census 显示每层 350 次 launch 中 85% 是 mHC：sinkhorn loop 162 次（48.6%）、pre/post mix + eager RMSNorm 122 次（36.6%），其余为 MoE/FFN、casts、attention、TP collective。按 correlation ID 归因 GPU time，Sinkhorn roll-up 占 kernel time 的 37.5%（每 decode step 6,966 launches），高于 `vllm::moe_forward_shared` 的 15.4%，`aten::sum` on `[6,4,4]` 单独占 21.5%。ITL 在 c=1→32 仅降 1.9x 而每 GPU 输出吞吐升 13x，说明设备远未饱和、per-step 成本由固定开销主导。

## 优化点

V4.1 的 delayed 公式 collapse 用上一 sublayer 的 pre-mix、并把当前 pre-mix 交给下一 seam，而 AITER `mhc_pre` 既不返回也不接受 pre-mix，无法直接复用；但对比显示 post_mix / comb_mix 在两种公式下 max_abs_err = 0（identical），仅 layer_input 因 stale coefficients 不同，昂贵部分（projection、RMS、gates 与全部 20 次 Sinkhorn，142 个 kernel 中的 141 个）bit-identical，只有 collapse 不同。本 PR 直接驱动 AITER 的两个 pre stage：`mhc_pre_gemm_sqrsum`（projection + row square-sums，1 kernel）与 `mhc_pre_big_fuse`（post/comb gates + 全部 Sinkhorn，1 kernel，其 collapse 被丢弃）；新增 `mhc_pre_mix_triton` epilogue 从同一 GEMM output 与 square-sums 恢复本 seam 的 pre-mix（不重做 projection，1 kernel），`hc_collapse_triton` 做带 carried coefficients 的 delayed collapse（1 kernel）。新 `MHCPreDelayedOp` 按 AITER → TileLang → reference 分派；model-entry broadcast seam 与 fused RMSNorm 两种情况 decline AITER 并 fall through 到 TileLang。模型侧三处 eager collapse（含 DSpark head）改走 `hc_collapse_triton`，`mhc_post` 调用点改走 `MHCPostOp`。

## 关键改动

```text
vllm/model_executor/kernels/mhc/aiter.py: 新增 MHCPreDelayedOp，组合 mhc_pre_gemm_sqrsum / mhc_pre_big_fuse 并按 AITER→TileLang→reference 分派
vllm/model_executor/kernels/mhc/triton.py: 新增 mhc_pre_mix_triton（从 unreduced GEMM 输出恢复 pre-mix）与 hc_collapse_triton（delayed collapse）
vllm/_aiter_ops.py: 暴露 AITER 的 mhc_pre_delayed 相关 op
vllm/models/deepseek_v4_1/amd/model.py: 把 eager Torch mHC / collapse / mhc_post 调用迁到 MHCPreDelayedOp、hc_collapse_triton、MHCPostOp
tests/kernels/test_mhc_kernels.py: 新增 AITER vs delayed reference 与 decline-unsupported 测试
```

## 性能声明

测试条件：gfx950 / MI355X（8x gfx950, TP4, DeepSeek-V4.1-Flash），`hidden_size=5120`、`hc_mult=4`、20 Sinkhorn iterations，对照 `mhc_pre_delayed_torch`；controlled A/B 在 CUDA graphs（`CUDAGraphMode.FULL_AND_PIECEWISE`）下同进程、相同输入张量、microseconds per call、median of 5 x 200 iterations；e2e 为 gsm8k 5-shot via lm-eval。launches per seam：tokens 1 / 6 / 24 / 96 / 1024 / 16384 均 141→4，worst rel err 2.4e-07→1.2e-05，collapse bit-exact，即 35x fewer launches per seam。Controlled A/B（torch / tilelang / aiter µs per call）：1 token 259.4 / 22.5 / **16.3**（15.89x vs torch、1.38x vs tilelang），96 tokens 383.3 / 31.3 / 18.0（21.30x、1.74x），128 tokens 391.4 / 30.4 / 18.8（20.87x、1.62x）。Timed eagerly 时 AITER 比 TileLang 慢（1 token 93.4 us vs 28.8 us，`_aiter_ops.mhc_pre_delayed` 的 Python wrapper 约 75 us host/call），但仍比 Torch reference 快 7.6x-8.2x。e2e gsm8k：baseline eager Torch mHC strict 0.9704±0.0047 / flexible 0.9697±0.0047 → AITER 0.9727±0.0045 / 0.9719±0.0045；nightly re-run 0.9719±0.0045 strict、0.9712±0.0046 flexible、1319/1319 problems。PR 注明端到端 ITL 数字来自 time-driven agentic trace、非 controlled A/B。

## 验证与回归风险

测试：`test_mhc_pre_delayed_rocm_aiter`（tokens 1/2/7/128/1024、carried 与 model-entry 两种 seam，断言 collapse bit-exact）与 `test_mhc_pre_delayed_rocm_aiter_declines_unsupported`（broadcast seam / fused norm 不静默走 AITER，带 positive control）。回归风险：#56214 的 `mhc_pre_big_fuse_with_norm_tilelang` 在 gfx950 + pinned tilelang 0.1.10 上有预存在 bug（15 个 fused_norm=True 用例失败，revert 本 PR 后 main+#56214 同样失败），不影响 serving（AMD decoder 单独应用 attn_norm/ffn_norm，不把 norm_weight 传入 pre block）；`enforce_eager` 用户有约 75 us/call 的 wrapper host 开销；AITER 路径仅 ROCm；被丢弃的 collapse 带来冗余 `[tokens, hidden_size]` bf16 store（decode 可忽略、约 1-2% prefill step）。

## 关联

#41946、#43950、#52737（DeepSeek-V4 迁到 MHCPreOp/MHCPostOp 并接 AITER）；#56214（TileLang fused-norm 预存在问题）；#56506（收集该模型 ROCm gaps 的 broader set）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
