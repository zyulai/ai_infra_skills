# PR #33255: [Bugfix] Fix quant RMS norm fusion for quantization with TMA-aligned scales

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ElizaWszola |
| 合并 | 2026-02-18 |
| 分类 | Quantization |
| Label | bug, performance, ready |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 含约 11 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/33255 |

## 一句话结论

修复 RMSNorm+per-token-group FP8 quant 融合在 TMA-aligned scales(scale 行按对齐补 pad)下 scale 索引用错 stride 的问题,使该融合在 DeepGEMM 场景恢复可用。

## 动机

#32619 引入 TMA-aligned scales 后,fused quant-layernorm 测试出现失败、融合匹配失效:融合 kernel 中 transposed scale 的行距被硬编码为 `gridDim.x`(token 数),而 TMA 对齐时实际分配的 scale 行数会向上取整到对齐边界,两者不一致导致 scale 读写错位;同时编译期 fusion pattern 无法匹配带 TMA 对齐 scale 的 quant op 签名(旧测试注释"on Hopper, tma-aligned scales inhibit matching")。

## 优化点

两层修复。Kernel 层:`rms_norm_per_block_quant` dispatch 把 `scales.stride(1)` 作为 `outer_scale_stride` 传入 device 代码,`compute_dynamic_per_token_scales` 与 `norm_and_quant` 中 transposed scale 的行索引从 `* gridDim.x` 改为 `* scale_rows`,其中 `scale_rows = ceil(gridDim.x / outer_scale_stride) * outer_scale_stride`,与分配时的 padded 行数一致;并新增 TORCH_CHECK 保证非 transposed 时 outer stride 必须为 1。融合层:`per_token_group_fp8_quant` op 增加 `dummy_is_scale_transposed`/`dummy_is_tma_aligned` 两个 dummy 参数(仅用于让 fusion pattern 按参数签名区分匹配),`rms_quant_fusion.py` 为每组 (group_shape, col_major, e8m0) 组合额外注册 `is_tma_aligned=True/False` 两套 pattern,scale 作为 pattern 输入传入而不是在 matcher 内部 make_scale;Python 侧 `rms_norm_per_block_quant` 用 `torch.empty_strided` 按 TMA 对齐分配 scales。

## 关键改动

```text
csrc/quantization/fused_kernels/layernorm_utils.cuh: scale 行距改为按 outer_scale_stride 对齐计算
csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_block_quant.cu: dispatch 传入 scales.stride(1)
csrc/quantization/w8a8/fp8/per_token_group_quant.cu + csrc/torch_bindings.cpp: quant op 增加 dummy 参数供融合匹配
vllm/compilation/passes/fusion/rms_quant_fusion.py: 注册 is_tma_aligned 两套 fusion pattern
vllm/_custom_ops.py: TMA 对齐时用 torch.empty_strided 分配 scales
tests/kernels/core/test_fused_quant_layernorm.py: 新增 TMA_ALIGNMENTS=[0,4] 参数化
```

核心 stride 修复:

```cuda
if constexpr (is_scale_transposed) {
  int64_t const scale_rows = (gridDim.x + outer_scale_stride - 1) /
                             outer_scale_stride * outer_scale_stride;
  all_token_scales[(threadIdx.x / threads_per_group) * scale_rows +
                   blockIdx.x] = scale;
}
```

## 性能声明

Benchmark(Hopper,`Qwen/Qwen3-30B-A3B-FP8`,单位 ms,main vs PR,18 组 input-len/prefix-len/output-len/request-rate 组合):TTFT 与 TPOT 总体互有涨跌、差异多在约 ±0.5%–2% 内;例如 rr=1、input 512 时 main_ttft 45.18 → pr 43.96、main_tpot 13.04 → 12.86;rr=10、input 1024 时 main_ttft 79.23 → 87.02(变差)。PR 正文未给出汇总加速比结论。

GSM8K 5-shot(Hopper,Qwen3-30B-A3B-FP8):本 PR With Deep Gemm flexible-extract 0.8597 / strict-match 0.8878,Main 分支 0.8590 / 0.8923;Without Deep Gemm 两者完全相同(0.8256 / 0.8946)。

## 验证与回归风险

测试:`tests/kernels/core/test_fused_quant_layernorm.py` 新增 TMA alignment 参数(0 与 4)覆盖融合 kernel 数值正确性;`tests/compile/fusions_e2e/test_tp1_quant.py` 重新启用 Hopper 上的 DeepGEMM 融合匹配测试(Blackwell 仍 skip)。风险面:改动触及 `per_token_group_fp8_quant` 的 op 签名(所有 CUDA 调用点已同步)与 fusion pattern 注册数量(组合数翻倍,编译期 pattern 数增多);scale 索引逻辑同时影响 vectorized 与非 vectorized 两条 device 路径,依赖参数化测试兜底。

## 关联

前序:#32619(引入 TMA-aligned scales,本 PR 修复其造成的融合失效)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
