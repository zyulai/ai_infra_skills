# PR #2881: Support w8a8 int8 quantization config

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2025-01-14 |
| 分类 | Quantization |
| Label | enhancement, high priority, performance, quant |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 4 个文件(1 新增 3 修改) |
| 链接 | https://github.com/sgl-project/sglang/pull/2881 |

## 一句话结论

新增 w8a8_int8 量化配置,走 sgl-kernel int8 GEMM,A100 上比 compressed-tensors 路径输出吞吐高约 10% 且精度无损。

## 动机

为 W8A8 int8(weight 静态 per-channel symmetric、activation 动态 per-token symmetric)提供官方量化配置,直接对接 sgl-kernel 已有的 `int8_gemm_kernel.cu`(int8 GEMM)与 `int8_kernel.py`(int8 量化 kernel)。此前 compressed-tensors 格式的 W8A8 模型走通用加载路径,本配置针对该量化方案使用更快的专用 GEMM。

## 优化点

新增 `W8A8Int8Config` / `W8A8Int8LinearMethod`:权重以 `ModelWeightParameter`(int8,`input_dim=1/output_dim=0`)+ `ChannelQuantScaleParameter` 注册,加载后转置;`apply` 时先 `per_token_quant_int8(x)` 动态量化激活,再调用 sgl-kernel 的 `int8_scaled_mm(x_q, weight, x_scale, weight_scale, out_dtype, bias)` 完成带反量化的 int8 GEMM。最低算力要求 75(Turing+),支持 fp16/bf16。配置接入侧:`model_config.py` 的 `_verify_quantization` 接受 `w8a8_int8`,并新增 `compatible_quantization_methods` 映射,允许 CLI 指定 `w8a8_int8` 加载 checkpoint 中标为 `compressed-tensors` 的模型(避免两者不一致时的报错);`server_args.py` 的 `--quantization` 候选加入 `w8a8_int8`。注意它不改变量化格式,只改变推理执行路径。

## 关键改动

```text
python/sglang/srt/layers/quantization/w8a8_int8.py: 新增 W8A8Int8Config 与 LinearMethod(apply 走 int8_scaled_mm)
python/sglang/srt/configs/model_config.py: 支持 w8a8_int8 并兼容 compressed-tensors checkpoint
python/sglang/srt/layers/quantization/__init__.py: 注册 w8a8_int8
python/sglang/srt/server_args.py: --quantization 增加 w8a8_int8
```

```python
def apply(self, layer, x, bias=None):
    x_q, x_scale = per_token_quant_int8(x)
    return int8_scaled_mm(
        x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias
    )
```

## 性能声明

A100,gsm8k `bench_sglang.py --num-questions 1400 --parallel 1400`,`--disable-radix`:

- Meta-Llama-3-8B-Instruct W8A8:compressed-tensors 2155.323 token/s(acc 0.735)→ w8a8_int8 2363.601 token/s(acc 0.738),latency 60.049 → 54.627 s
- Qwen2-7B-Instruct W8A8:2959.476 token/s(acc 0.820)→ 3254.543 token/s(acc 0.827),latency 61.788 → 56.783 s
- 作者结论:输出吞吐高约 10% 且无精度损失(两次对比精度均略升);Invalid 率不变(0.002 / 0.001)

## 验证与回归风险

PR 正文未注明新增自动化测试,验证为上述两个模型的 gsm8k 精度+吞吐对比。兼容范围:仅 HF per-channel symmetric int8 checkpoint。风险:`get_quant_method` 直接 `from vllm.model_executor.layers.linear import LinearBase`,引入对 vllm 包的运行时依赖;int8 GEMM 路径依赖 sgl-kernel CUDA kernel(非 CUDA 不可用);weight 转置后的内存布局与 per-channel scale 语义依赖 checkpoint 格式正确。

## 关联

同分类:其余 quantization 配置(fp8、modelopt 等)并列注册于 `QUANTIZATION_METHODS`;复用的 sgl-kernel int8 GEMM 与 `int8_kernel.py` 均为本 PR 之前已存在(编号素材未注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
