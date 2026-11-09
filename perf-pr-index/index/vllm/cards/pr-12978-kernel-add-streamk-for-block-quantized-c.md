# PR #12978: [Kernel]Add streamK for block-quantized CUTLASS kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Hongbosherlock |
| 合并 | 2025-02-21 |
| 分类 | Quantization |
| Label | ready |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/12978 |

## 一句话结论

block-quantized FP8 CUTLASS GEMM 在 K 很大时切换 StreamK tile scheduler:DeepSeek-V3 形状下 28.5→17.1 µs,文中称最高 60%+ 提升。

## 动机

blockwise scale(scale_a 1x128 / scale_b 128x128,DeepSeek V3 用法)的 CUTLASS FP8 scaled mm 在 K 很大时明显变慢,典型如 DeepSeek V3 中的 `m128-n1536-k7168`:K 大时默认 PersistentScheduler 下各 SM 工作量不均、尾部的长 K tile 拖慢整体。作者基于 #11868 探索 Stream-K 与 Split-K 两种分解策略,结论是 Stream-K 在多数场景更优。

## 优化点

`cutlass_gemm_caller` 新增 `TileSchedulerArguments` 形参并附带 `KernelHardwareInfo`;`cutlass_3x_gemm_fp8_blockwise` 模板新增 `SchedulerType` 参数,`KernelType` 由硬编码 `PersistentScheduler` 改为传入的调度器。当 `TileSchedulerTag` 是 `StreamKScheduler` 时,设置 `decomposition_mode = StreamK`、`reduction_mode = Nondeterministic`(非确定性归约)。dispatch 层加形状启发式:`K > 3N` 时实例化 StreamKScheduler 版本,否则保留 PersistentScheduler,保证非 large-K 场景零影响。

## 关键改动

```text
csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh: caller 接受 tile scheduler 参数
csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh: SchedulerType 模板参数与 K>3N 分派
```

```cpp
auto k = a.size(1);
auto n = b.size(1);

if (k > 3 * n) {
  cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<
      cutlass::gemm::StreamKScheduler, OutType, 1, 128, 128>>(
      out, a, b, a_scales, b_scales);
} else {
  cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<
      cutlass::gemm::PersistentScheduler, OutType, 1, 128, 128>>(
      out, a, b, a_scales, b_scales);
}
```

## 性能声明

- 测试条件:DeepSeek-V3 的 GEMM 形状,H800,TP-4,`scaled-torch.float8_e4m3fn-gemm`(blockwise scaled mm benchmark,时间单位 µs,越低越好)。
- large-K 形状显著加速:`MKN=(128x7168x1536)` 28.5 → 17.1 µs,`MKN=(128x7168x1024)` 26.4 → 17 µs,`MKN=(256x7168x2048)` 29.8 → 22 µs;其余非 large-K 形状持平或 ±0.4 µs 内。
- PR 正文声明 Stream-K 相比 baseline 多数场景超过 60% performance improvement(表格中最大单形状改善约 40%)。

## 验证与回归风险

diff 未包含测试,PR 正文未附验证/回归计划。风险面:`K > 3N` 启发式对全部 blockwise FP8 GEMM(sm90 路径)生效,包括非 DeepSeek 模型;StreamK 的 `ReductionMode::Nondeterministic` 使输出非位可复现(相对原 PersistentScheduler 是数值语义变化);仅覆盖 scale_a 1x128 / scale_b 128x128 的 block 形状。

## 关联

基于 #11868(blockwise CUTLASS scaled mm 的前序工作)。索引同分类另有 pr-24757(float→FP8 硬件量化指令)、pr-18844(scaled FP8 quant 调参)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
