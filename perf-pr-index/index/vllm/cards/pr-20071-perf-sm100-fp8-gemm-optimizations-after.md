# PR #20071: [Perf] SM100 FP8 GEMM Optimizations after cutlass_profiler

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-06-27 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 1 个文件(scaled_mm_sm100_fp8_dispatch.cuh) |
| 链接 | https://github.com/vllm-project/vllm/pull/20071 |

## 一句话结论

用 cutlass_profiler 重调 SM100 FP8 GEMM 的 M<=256 配置,部分 NxK 提速 1.12x~1.5x。

## 动机

PR 正文:这是 #19566 之后的追加性能优化。借助 cutlass_profiler 的结论,针对 M <= 256 重新调优 CUTLASS 配置,目标是在不影响其他 NxK 性能的前提下,把某些 NxK 对从 1.12x 提升到 1.5x speedup。正文说明选取标准是「对静态 M,在多个 NxK 对上综合表现最好的配置」(给出 cta_m/cta_n/cta_k、cluster_m/cluster_n/cluster_k 的 Rank 与 GFLOPs 表格)。

## 优化点

改动集中在 `scaled_mm_sm100_fp8_dispatch.cuh` 的按 M 分档与 tile/cluster 配置:把分档边界由 [1,64] / (64,128] / (128,256] 改为 [1,16] / (16,64] / (64,256],`mp2` 下界从 64 改为 16,并新增 `sm100_fp8_config_M16`、把原 M128/M64 重命名重排。`sm100_fp8_config_M256` 的 ClusterShape 由 `Shape<_2,_2,_1>` 改为 `Shape<_2,_1,_1>`;新的 M64 档用 `TileShape<64,64,128>` + `ClusterShape<1,1,1>`;新的 M16 档用 `TileShape<64,64,128>` + `ClusterShape<1,4,1>`(原 M64 为 `<64,64,256>` + `<1,8,1>`)。即小 M 采用更小的 cluster 与 cta_k。

## 关键改动

```text
csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh: 重排按 M 分档([1,16]/(16,64]/(64,256])、新增 M16 配置、改 M256/M64 的 TileShape 与 ClusterShape
```

关键机制代码(dispatch 边界):

```cpp
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));  // next power of 2
  if (mp2 <= 16) {
    return cutlass_gemm_caller<Cutlass3xGemmM16>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    // m in (16, 64]
    return cutlass_gemm_caller<Cutlass3xGemmM64>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 256) {
    // m in (64, 256]
    return cutlass_gemm_caller<Cutlass3xGemmM256>(
```

## 性能声明

PR 正文 Kernel benchmarks 使用 #17126 在 B200 上跑 `python benchmarks/kernels/bench_fp8_gemm.py --model meta-llama/Llama-3.1-8B-Instruct --tp-sizes 1` 与 `--model meta-llama/Llama-3.3-70B-Instruct --tp-sizes 4`,对比 fp8-tensor-w-tensor-a-noquant 的 Before/After(TFLOP/s)与 speedup。最大增益:Llama-3.3-70B、N=8192 K=7168,batch_size=1 时 9.84 → 12.38(1.26)、16 时 156.65 → 199.27(1.27)、32 时 309.39 → 451.10(1.46)、64 时 608.19 → 938.37(1.54)、128 时 1158.40 → 1777.43(1.53);同模型 N=8192 K=2048 的 batch=64 为 345.92 → 467.90(1.35)、128 为 654.82 → 861.29(1.31)。正文总述 "from 1.12x to 1.5x speedup for certain NxK pairs without affecting performance for the other NxK",其余 NxK 组合 speedup 多在 0.93~1.09。PR 未注明精度参数(workload 为 FP8 GEMM kernel benchmark)。

## 验证与回归风险

PR 依赖 `benchmarks/kernels/bench_fp8_gemm.py` 的 kernel benchmark 与 #17126 的对比框架,正文未列出新增单元测试。影响路径:SM100(B200)上走 `cutlass_gemm_sm100_fp8_dispatch` 且 M<=256 的 FP8 W8A8 GEMM;由于分档边界下移(64→16)且各档 tile/cluster 变化,原本落在旧 (64,128] 档的形状会改用新配置。正文自述对「其他 NxK」性能无影响,但部分组合 speedup 略低于 1(最低 0.93),属需关注的回归点。

## 关联

前序 PR #19566(正文标注为 "Additional performance optimizations after #19566");kernel benchmark 工具来自 #17126(正文所引)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
