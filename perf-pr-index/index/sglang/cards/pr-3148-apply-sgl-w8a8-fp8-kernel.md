# PR #3148: Apply sgl w8a8 fp8 kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HandH1998 |
| 合并 | 2025-03-09 |
| 分类 | Quantization(tier: major) |
| Label | high priority, quant |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 13 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/3148 |

## 一句话结论

w8a8 FP8 GEMM 从 vllm kernel 切换到 sgl-kernel,sm89-L40 吞吐 +14% 且 gsm8k 精度更高。

## 动机

接续 #3047,把 w8a8 fp8 linear 的 GEMM kernel 从 vllm 实现替换为 sgl-kernel。PR 正文给出两点依据:sgl-kernel 的 w8a8 fp8 在 gsm8k 上精度普遍更高;在 sm89-L40 上吞吐比 vllm kernel 高 14%,而在 sm90-H100 上两者相近。当时 SGLang 的 `apply_fp8_linear` 直接复用 vllm `_custom_ops`,kernel 与量化布局(per-tensor/per-channel)耦合在 vllm 代码上,值得本地化。

## 优化点

新增 `w8a8_fp8` 量化方案(W8A8Fp8Config / W8A8Fp8LinearMethod:weight 静态 per-channel、activation 动态 per-token,min capability 89)。把 `apply_fp8_linear` 移入 `fp8_utils.py`:cutlass 可用时调用 sgl-kernel 的 `fp8_scaled_mm`(per-channel weight scale + per-token activation scale),保留 `USE_VLLM_CUTLASS_W8A8_FP8_KERNEL` 环境变量回退 vllm cutlass kernel;不支持 cutlass 时回退 `torch._scaled_mm` per-tensor 或 naive dequant 路径。新增 `static_quant_fp8`(Triton,把 per-tensor static scale 广播为 per-token)并绑定 `sgl_per_token_quant_fp8`。`fp8.py` 的 `process_weights_after_loading` 在 cutlass/marlin 路径默认做 per-channel 权重量化;`BlockQuantScaleParameter` 上移到 `parameter.py`;`modelopt_quant` 改用本地 `apply_fp8_linear` 并在 cutlass 下转 channelwise scale。

## 关键改动

```text
python/sglang/srt/layers/quantization/w8a8_fp8.py: 新增 W8A8Fp8Config/W8A8Fp8LinearMethod
python/sglang/srt/layers/quantization/fp8_utils.py: 本地 apply_fp8_linear,接入 sgl-kernel fp8_scaled_mm
python/sglang/srt/layers/quantization/fp8_kernel.py: 新增 sglang_per_token_quant_fp8 / static_quant_fp8
python/sglang/srt/layers/quantization/fp8.py: cutlass/marlin 分支改用 per-channel 权重量化
```

## 性能声明

模型 neuralmagic/Meta-Llama-3-8B-Instruct-FP8:sm89-L40 gsm8k(2000 题)输出吞吐 616.437 → 702.757 token/s("14% higher")、acc 0.752 → 0.761;各 request rate 吞吐(rate=inf)2122.10 → 2422.71 tok/s。sm90-H100 相近(2801.533 → 2805.524 token/s,acc 0.754 → 0.759)。Meta-Llama-3.1-8B-Instruct-FP8-dynamic:L40 acc 0.779 → 0.795、输出吞吐 590.980 → 694.344 token/s;H100 acc 0.754 → 0.796、latency 49.751 → 47.991 s。

## 验证与回归风险

新增 `TestStaticQuantFP8` 单测(static_quant_fp8 对照 native 实现,多 dtype/shape),并把 per_token_group_quant 测试的 rtol 从 0.15 放宽到 0.20。回归面:`fp8.py` / `modelopt_quant.py` 权重加载路径改动影响所有 FP8 checkpoint(cutlass/marlin 下权重布局变为 per-channel);ROCm 仍走 e4m3fnuz 归一化分支。

## 关联

前序 PR #3047;同分类(Quantization)另有 PR #28662(DeepGEMM UE8M0 scale 处理)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
