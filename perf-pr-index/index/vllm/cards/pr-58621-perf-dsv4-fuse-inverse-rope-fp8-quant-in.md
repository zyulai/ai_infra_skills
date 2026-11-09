# PR #58621: [Perf][DSv4] Fuse inverse RoPE + FP8 quant into FlashInfer sparse MLA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-09-25 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, deepseek, nvidia, quantization, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾标注 2 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/58621 |

## 一句话结论

RopeQuant 把 inverse RoPE+FP8 融进 sparse MLA，8k/1k serving 吞吐 +6.5~16%、TPOT 改善 6–9%。

## 动机

FlashInfer 0.7.0（已 pin）自带 TRTLLM-GEN DSv4 sparse-MLA 的 RopeQuant epilogue（flashinfer#4918）：kernel 对 attention 输出做 inverse RoPE 并转 FP8（带 packed UE8M0 scales，布局正是 DeepGEMM `fp8_einsum` 所需），但 vLLM 尚未接入。原路径要写出 bf16 `[T, 128, 512]` attention 输出再单独跑一遍 `fused_inv_rope_fp8_quant`，prefill 需写又重读大块 buffer。此外 DSpark non-causal decode 的 index 布局存在 bug：`-1` padding 被当作 key、上下文少于 128 token 时因果窗口切掉块内后续 token。

## 优化点

`DeepseekV4Attention` 增 `_alloc_attn_out` hook（默认仍是 padded bf16 buffer，FlashMLA/SM120/ROCm/XPU/CPU 层不变）；`DeepseekV4FlashInferMLAAttention` 分配 RopeQuant buffers，`deep_gemm_fp8_o_proj` 现在也接受 `QuantizedActivation`，跳过 `fused_inv_rope_fp8_quant`、把 kernel 的 FP8 输出与 scales 直送 `fp8_einsum`。launcher 要求输出 group stride 等于 `sum_q`，故 decode/prefill 均跨完整 CUDA-graph-padded buffer、用 `cum_seq_lens_q` 选 token。`build_flashinfer_mixed_sparse_indices` 增 `num_rows`，并修复 DSpark non-causal 布局（causal window 填首个可见项、其余自列 128 起、`sparse_topk_lens` 精确设置）。自动启用（无 flag）需恰好 128 local query heads（DSv4-Pro TP1 或 DP attention）、per-tensor FP8 KV cache、FP8 `wo_a`，其余保持原路径。

## 关键改动

```text
vllm/models/deepseek_v4/attention.py: 新增 _alloc_attn_out hook，默认返回 padded bf16 buffer，仅 FlashInfer 子类改分配 RopeQuant buffers
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py: deep_gemm_fp8_o_proj 接受 QuantizedActivation，跳过 fused_inv_rope_fp8_quant，把 kernel FP8 输出与 scales 直送 fp8_einsum
vllm/models/deepseek_v4/common/ops/cache_utils.py: build_flashinfer_mixed_sparse_indices 增 num_rows，并修复 DSpark non-causal decode 行布局
tests/kernels/attention/test_flashmla_sparse.py: 新增 fused/unfused wo_a 一致性与 non-causal 行、pad 行布局单测
```

## 性能声明

Serving perf（8k in/1k out，DeepSeek-V4-Pro-0813，DEP4 单 GB300 节点、4 DP ranks tp1 故 128 local heads，`FLASHINFER_MLA_SPARSE_DSV4`、FP8 KV、MegaMoE、无投机解码，vllm-bench 闭环 sweep）：output tok/s/GPU fused/base 为 c16 213.3/183.8（+16.0%）、c64 444.7/417.5（+6.5%）、c256 853.3/783.3（+8.9%）；正文总结 TPOT 改善 6–9%、吞吐 6.5–16%；ITL p99 c64 68.0 vs 490.5 ms；c1–c4 的 decode 慢 2–2.5%。Kernel time（µs，128 heads、decode q_len=1）：MTP(q_len=2) 下 fused 1.01–2.13× faster，单 token decode、batch≤8、长 KV 略慢约 1 µs/C4A 层。E2E with DSpark：acceptance length gsm8k 4.42 vs 3.70（+19.5%）、aime25 3.83 vs 3.31（+15.9%）；gsm8k/aime25 精度在噪声内持平。

## 验证与回归风险

Tests 全部在 GB300 通过：新单测 `test_flashinfer_rope_quant_matches_unfused_o_proj`、`test_flashinfer_dspark_noncausal_block_sees_future_tokens`；现有 `test_flashmla_sparse` 19/19、`test_dspark_noncausal_sparse_mla` 33 passed 8 skipped、`test_deepseek_v4_swa_visible` 28/28、`test_fused_inv_rope_fp8_quant` 82/82；pre-commit 含 mypy 通过。回归风险：仅 128 local heads + FP8 KV + FP8 wo_a 时启用，Pro TP2–16、Flash、SM120/ROCm/XPU/CPU 保持原路径；draft block 超过 `max_model_len` 时位置 clamp 不被 RopeQuant 复现（仅影响这些 draft）；低并发 decode 略慢；DSv4.1 的 MXFP8 `wo_a` 需 per-32 scales、而 shipped RopeQuant 只发 per-128，列为 follow-up 未纳入本 PR。

## 关联

正文明确 #57432（DSv4.1 同款 non-causal 布局修复，本 PR 为 DSv4 对应修复、做法不同）、#57679（query RMSNorm+MXFP8）、#57977（DeepGEMM 间 MXFP8 activations）、#55459（indexer Hadamard），以及 flashinfer#4918（RopeQuant epilogue）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
