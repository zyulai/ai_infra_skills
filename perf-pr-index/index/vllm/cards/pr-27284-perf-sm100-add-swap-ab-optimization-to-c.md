# PR #27284: [Perf] SM100 - add swap AB optimization to CUTLASS FP8 GEMM

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LyrisZhong |
| 合并 | 2025-11-04 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `scaled_mm_sm100_fp8.cu` 与 `scaled_mm_sm100_fp8_dispatch.cuh`(CUDA/CUTLASS) |
| 链接 | https://github.com/vllm-project/vllm/pull/27284 |

## 一句话结论

B200 (SM100) CUTLASS FP8 GEMM 对 M≤64 启用 swap-AB 并按 nvMatmulHeuristics 调参,Llama 450B 各形状提速 5-25%。

## 动机

SM100 第五代 Tensor Core(`tcgen05.mma`)仅支持固定 M=64×N×16(N 为 8 的倍数)和 M=128×N×16(N 为 16 的倍数)的矩阵形状。当 batch 维 M 很小(≤64)时,把 A 的 M 维 padding 到 64/128 浪费算力;交换 M/N 后 padding 负担转移到更容易对齐的 N 维上,可显著减少小 batch GEMM 的 padding 开销。H200 上已有同类优化(#20396),SM100 的 FP8 路径此前缺失。

## 优化点

新增 `cutlass_3x_gemm_sm100_fp8` 模板,带编译期 `swap_ab` 标志:swap 时 mainloop 把 B(按转置 layout)当 A、A 当 B,prob_shape 变 `(n, m, k, 1)`,epilogue 相应采用转置的 C/D layout,bias 场景改用 `ScaledEpilogueColumnBias`(列向量 bias 匹配转置后的输出)。dispatch 按 M 分档重写:原先按 next_pow_2(M) 分四档,现在 `m<=16` 与 `m<=64` 走 swap-AB 配置(`sm100_fp8_config_M16_swap_ab`,TileShape 128×32×128、Cluster 4×1×1;`sm100_fp8_config_M64_swap_ab`,TileShape 128×64×256、Cluster 4×1×1),但 `m==64 且 k<4096` 例外地走非 swap 配置(实测 swap 反而不利);更大 M 保持原档位。各档配置均借助 nvMatmulHeuristics 调优。调用方接口从 Epilogue 模板参数改为 `EnableBias` bool 并显式传 a_scales/b_scales。

## 关键改动

```text
csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8_dispatch.cuh:
  新增 swap_ab 模板 Gemm、swap 感知的 gemm_caller、M16/M64 swap-ab
  配置与 M 分档 dispatch
csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu:
  入口改为 <true/false> EnableBias 模板并传递 scales
```

关键片段(dispatch):

```cpp
if (m <= 16) {
  return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM16SwapAB>(
      out, a, b, b_scales, a_scales, ...);
} else if (m <= 64) {
  if (m == 64 && k < 4096) { /* do not enable swap AB */ }
  return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM64SwapAB>(...);
}
```

## 性能声明

Llama 450B 问题空间(8×B200,TP8),`benchmarks/kernels/bench_fp8_gemm.py`:跨不同 M/N/K 与量化方案提升 5% 至 25%;详细数据以截图给出(PR 正文未列数字表)。Llama 70B 也在测试范围内但正文未单独列数。精度:`tests/kernels/quantization/test_cutlass_scaled_mm.py` 的 `test_cutlass_fp8_gemm` 152 passed、`test_cutlass_fp8_gemm_m_sweep` 8 passed。

## 验证与回归风险

验证:CUTLASS FP8 GEMM 单元测试全量通过(152 + 8 个用例),性能基准覆盖 Llama 450B/70B 的 GEMM 问题空间。风险:swap-AB 仅作用于 M≤64 的小 batch decode 场景,大 M 路径代码等价重构(Epilogue 模板参数改为 EnableBias),bias 路径在 swap 时必须用 ColumnBias 变体,若未来新增 epilogue 类型需注意;`m==64 && k<4096` 的豁免是经验性阈值,其他 K 值未在 PR 中给出解释。

## 关联

同类优化前序:#20396(H200 swap-AB)。同 Quantization 分类:#42235(FlashInfer cutedsl NVFP4)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
