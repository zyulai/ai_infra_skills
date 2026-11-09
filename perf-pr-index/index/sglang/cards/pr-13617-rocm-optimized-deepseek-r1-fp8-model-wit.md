# PR #13617: [ROCM] Optimized deepseek-r1 fp8 model with + triton_gemm_a8w8 + batch_gemm_a8w8 + fused set_mla_kv_buffer kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yctseng0211(co-author @yichiche) |
| 合并 | 2025-11-20 |
| 分类 | Quantization(tier: major) |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/13617 |

## 一句话结论

MI355X 上 DeepSeek-R1 FP8 的 a8w8 GEMM 换 Triton 实现、启用 batched GEMM 与 fused set_mla_kv_buffer,w/o torch compile 性能提升 4–20%。

## 动机

优化 DeepSeek-R1 在 ROCM 上的性能。Profiling 显示 aiter 的 `gemm_a8w8_blockscale` 在 concurrency 1–8 区间延迟不如 Triton 实现,且换用 Triton 版能显著增加 elementwise–GEMM 融合的机会。同时 MI355X 上 MLA 的 q_nope/w_kc、attn_output/w_vc 两个 bmm 走的是"上转 bf16 + torch.bmm"路径,fp8 权重的能力没有被利用;MLA KV buffer 的 fused set/get 路径此前只对 CUDA 开放。

## 优化点

三项改动,均限定在 ROCm/aiter 路径:(1) `fp8_utils.py` 中把 aiter 的 `gemm_a8w8_blockscale`(CUDA/汇编版)替换为 `aiter.ops.triton.gemm_a8w8_blockscale` 的 Triton 实现;(2) gfx95(MI355X)上启用 `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`,在 `deepseek_v2.py` 的 forward_absorb_prepare/forward_absorb_core/forward_npu_sparse_prepare 三处,当 `_use_aiter_gfx95` 且 `w_kc/w_vc` 为 `float8_e4m3fn` 时,用它替代"bf16 上转换 + torch.bmm",直接做 per-token-group 量化的 batched fp8 GEMM;(3) 把 `_set_mla_kv_buffer`/`_get_mla_kv_buffer` 的 fused 路径条件从 `_is_cuda` 扩展为 `_is_cuda or _use_aiter_gfx95`,使 ROCm 上也走 fused MLA KV buffer 读写。所有分支以平台与权重 dtype 为条件,其余路径不变。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_utils.py: a8w8 blockscale GEMM 改用 aiter 的 Triton 实现
python/sglang/srt/models/deepseek_v2.py: MLA absorb 路径接入 batched fp8 GEMM 与 fused set/get_mla_kv_buffer
```

```python
if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        X=q_nope,
        WQ=self.w_kc.transpose(-1, -2),
        w_scale=self.w_scale,
        group_size=128,
        YQ=None,  # allocate (B, M, N)
        transpose_bm=False,  # (B, M, N)
        transpose_bm_in=True,  # (M, B, K)
        dtype=torch.bfloat16,
    )
```

## 性能声明

DeepSeek-R1-0528,TP8,MI355x(`SGLANG_USE_AITER=1`),server mode(chunked-prefill 196608、cuda-graph-max-bs 128、num-continuous-decode-steps 4):w/ torch compile(torch compile max_bs = 32)性能提升 3–10%;w/o torch compile 性能提升 4–20%。正文未给出 workload 的 ISL/OSL/并发设定与分项数字。accuracy:gsm8k(--num-questions 2000 --parallel 2000)改前 0.93、改后 0.93。

## 验证与回归风险

accuracy 验证仅 gsm8k 2000 题前后对比;checklist 中"Add unit tests"未勾选,diff 未新增测试。回归面:`_use_aiter_gfx95` 且 fp8 权重时的 MLA bmm 与 KV buffer 路径;非 gfx95、非 fp8 权重与 CUDA 路径保持原实现。pack 未包含 CI States 记录。

## 关联

正文未引用其他 PR 编号;同批索引中 #35719、#36505、#36527 亦属 AMD/ROCm 优化区域。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
