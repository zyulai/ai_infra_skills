# PR #55170: [Perf][Quant][NVFP4] Prefer W4A4 linear kernels over weight-only ones on SM120/121

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stecasta |
| 合并 | 2026-09-08 |
| 分类 | Quantization |
| Label | quantization |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/55170 |

## 一句话结论

SM120/121 上调整 NVFP4 linear kernel 选择顺序,让 W4A4 kernel 优先于 weight-only CuTe-DSL kernel,Qwen3.8-27B-NVFP4 输出吞吐 +23.8%、TTFT -36%。

## 动机

SM120/121 上 NVFP4 linear 层即使 checkpoint 能提供 FP4 activation,也会选到 weight-only(W4A16)kernel,FP4 tensor core 闲置。原因:`_POSSIBLE_NVFP4_KERNELS[CUDA]` 按序取第一个报告支持的 kernel,首位 `FlashInferCuteDslNvFp4LinearKernel`(W4A4)被 gate 到 sm_10x(FlashInfer CuTe-DSL FP4 GEMM 只带 sm100/sm103 tactic selector),在 SM120 上被跳过,选择落到第 2 位的 weight-only kernel,排在三个可用的 W4A4 kernel 之前。该顺序来自 #53014(其 forced weight-only 路径 gate 了 SM100/103,但优先级列表项未 gate)。

## 优化点

纯顺序修复:把 `FlashInferCuteDslNvFp4W4A16LinearKernel` 移到三个 W4A4 kernel(`FlashInferCutlassNvFp4LinearKernel`、`FlashInferB12xNvFp4LinearKernel`、`CutlassNvFp4LinearKernel`)之后、`MarlinNvFp4LinearKernel` 旁边。weight-only checkpoint 不受影响:`use_a16` 只会把候选列表收窄到 weight-only kernel 并保持其相对顺序,forced weight-only 路径本就按 compute capability 显式选择。kernel 选择是每层静态的,因此取的是 mixed prefill+decode workload 的更优默认;W4A4 在 prefill 大幅领先、BS1 decode 略输几个百分点,decode-only workload 更适合 weight-only kernel。按 forward 动态选择需要两种 kernel 共享 weight layout,依赖 #54614。

## 关键改动

```text
vllm/model_executor/kernels/linear/__init__.py: _POSSIBLE_NVFP4_KERNELS[CUDA] 中 W4A16 项移到 W4A4 项之后
tests/kernels/quantization/test_nvfp4_kernel_selection.py: 新增顺序断言测试(CPU-only)
```

```python
PlatformEnum.CUDA: [
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    CutlassNvFp4LinearKernel,
    # Weight-only: a fallback on a W4A4-capable checkpoint, not a preference.
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    MarlinNvFp4LinearKernel,
    ...
]
```

## 性能声明

Qwen3.8-27B-NVFP4,RTX PRO 6000 Blackwell Max-Q,BS1,32K cached prefix + 2K ISL + 256 OSL,同节点同 checkpoint,`v0.28.1rc1.dev337+g27a94d1ce` + V2 model runner:`auto`(选中 `FlashInferCuteDslNvFp4W4A16LinearKernel`)TTFT 1455.2 ms、74.81 tok/s;`flashinfer_cutlass`(选中 `FlashInferCutlassNvFp4LinearKernel`)TTFT 926.1 ms、92.63 tok/s。+23.8% 输出吞吐、TTFT 低 36%。V1 runner 上同样对比 +18.7%;V1 下 `marlin` 为 74.60 tok/s,weight-only CuTe-DSL 为 73.08(SM120/121 上 weight-only 也不是最优)。差距集中在 prefill。硬件/精度细节如上,其余 PR 未注明。

## 验证与回归风险

新增 `test_w4a16_kernel_does_not_precede_w4a4_kernels`:断言 CuTe-DSL W4A16 kernel 不排在三个 SM120/121 可用 W4A4 kernel 之前,main 上三个都失败、本修复后通过,CPU-only 无需 GPU。SM103(GB300)验证为 inert:`auto` 在首位就选中 W4A4 CuTe-DSL kernel,被移动项不可达;weight-only checkpoint 在 SM103 仍选 W4A16 kernel。风险:静态 per-layer 选择对 decode-only workload 略有回退(BS1 decode W4A4 输几个百分点),注释已说明;其他平台与 kernel 支持面的变化不在本 PR 范围。

## 关联

#53014(引入该顺序的前序 PR)、#54614(共享 weight layout 以支持 per-forward 选择)、#55103(Marlin scale-factor 内存)、#55073(MoE w13 global scales)——同区域但表面不同。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
