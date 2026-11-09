# PR #4362: Fix per token fp8 quant precision

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4362](https://github.com/sgl-project/sglang/pull/4362) |
| 作者 | qingquansong |
| 合并日期 | 2025-03-13 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
per-token fp8 量化在小输入下输出有精度问题;去掉额外的 1/block_max 倒数换算,直接用 block_max 参与量化修复。

## 问题现象
per-token 量化在小输入尺寸(#token x hidden_dim 较小)时存在精度问题:正文明确「Scale is computed correctly but quantized output has precision issue for small input sizes」,即 scale 本身计算正确,但量化后的输出在输入较小时有精度偏差。测试中的对比基准为 vLLM 的 per-token fp8 量化实现,以 `torch.allclose`(out: rtol=1e-3, atol=1e-5)校验。

## 根因
正文披露:量化输出计算中多做了一次额外的倒数计算 `const float scale_val = 1.0f / block_max;`,再用乘法 `input * scale_val` 完成缩放。正文未进一步解释机理;(由 diff 推断)先取 fp32 倒数再乘回相当于引入两次浮点舍入,与直接除法 `input / block_max` 的舍入路径不同,在小输入、元素少的归约结果上舍入误差更易显化为可观测的量化输出偏差。

## 修复方案
1. `sgl-kernel/csrc/gemm/per_token_quant_fp8.cu` `per_token_quant_fp8_kernel`:删除 `const float scale_val = 1.0f / block_max;`,量化循环中由 `static_cast<float>(input_vec[j]) * scale_val` 改为 `static_cast<float>(input_vec[j]) / block_max`,直接除以 block_max 后做 ±FP8_E4M3_MAX 截断。
2. `sgl-kernel/benchmark/bench_per_token_quant_fp8.py` 与 `sgl-kernel/tests/test_per_token_quant_fp8.py`:scale 占位张量形状由 1D `[num_tokens]` 改为 2D `[num_tokens, 1]`,与输出形状对齐、免去事后 reshape;测试的 scale 断言容差由 atol=1e-3 收紧为 atol=1e-5,参数化网格从 3x3(128-512)扩为 5x5(含 32/64 token、128/256 hidden 的小输入组合)。

```diff
-  const float scale_val = 1.0f / block_max;
   ...
-      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
+      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) / block_max, FP8_E4M3_MAX), -FP8_E4M3_MAX);
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:sgl-kernel 的 `sgl_per_token_quant_fp8` CUDA 内核(所有 per-token 动态 fp8 量化调用方)输出数值与直接除法一致;除法与乘倒数吞吐相当,正文基准显示无速度回退(「Both diff check in benchmark and test cases passed and no speed regression」,表中 SGL Kernel 带宽在各 batch 下低于 VLLM 为既有差距);测试/基准脚本 scale 形状改为 2D,调用该 kernel 的下游需适配 shape 约定。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
