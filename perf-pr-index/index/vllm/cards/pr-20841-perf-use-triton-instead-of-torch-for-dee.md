# PR #20841: [Perf] Use Triton instead of Torch for DeepGEMM Per Token Group Quant

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-07-13 |
| 分类 | Quantization(major) |
| Label | performance, ready, deepseek |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 内容可用(5 个源码/测试文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/20841 |

## 一句话结论

把 DeepGEMM MoE 路径的 per-token-group FP8 量化从 DeepGEMM 自带的 Torch 实现切到 vLLM 的 Triton kernel,Qwen3-30B-A3B-FP8 单卡吞吐 20.70 → 23.85 req/s。

## 动机

本 PR 是 #20087 的 follow-up。此前 `vllm/utils/deep_gemm.py` 的 `per_token_group_cast_to_fp8` 在 Blackwell 且 DeepGEMM 提供 `per_token_cast_to_fp8` 新 API 时走 DeepGEMM 自己的 Torch 量化实现,否则才落到 vLLM 的 Triton kernel `per_token_group_quant_fp8`。同一功能两条实现并存,DeepGEMM 路径被绑在较慢的 Torch 量化上。

## 优化点

删除 `per_token_group_cast_to_fp8` wrapper 及 `deep_gemm.utils.math.per_token_cast_to_fp8` 的解析逻辑(`__all__` 同步移除);DeepGEMM MoE(`deep_gemm_moe.py` 的 a2 量化)与 fused_moe 通用入口(`fused_moe/utils.py` 的 `_fp8_quantize`,移除 `is_blackwell_deep_gemm_used()` 分支)一律调用 vLLM Triton kernel。同时为该 kernel 增加 `use_ue8m0` 编译期开关:当 Blackwell 上使用 DeepGEMM 时,scale 按 UE8M0 约定向上取整到 2 的幂,保持与 DeepGEMM scale 格式兼容;非 Blackwell 行为不变。row-major 与 column-major scales 两个 Triton 变体同步修改。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/fp8_utils.py: 两个 Triton 量化 kernel 增加 use_ue8m0 分支,调用点传入 is_blackwell_deep_gemm_used()
vllm/utils/deep_gemm.py: 删除 per_token_group_cast_to_fp8 wrapper 与 _per_token_cast_impl
vllm/model_executor/layers/fused_moe/deep_gemm_moe.py: a2 量化改调 per_token_group_quant_fp8
vllm/model_executor/layers/fused_moe/utils.py: _fp8_quantize 移除 Blackwell 分支
tests/kernels/moe/test_deepgemm.py、tests/kernels/quantization/test_block_fp8.py: 改用统一入口
```

```python
scale_raw = _absmax / fp8_max
y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
```

## 性能声明

PR 正文(Qwen3 单卡,`VLLM_USE_DEEP_GEMM=1 vllm bench throughput --model Qwen/Qwen3-30B-A3B-FP8 --load-format dummy --input-len 1000 --output-len 100 --enforce-eager --enable-expert-parallel`):修改后 23.85 requests/s、26179.47 total tokens/s、2384.89 output tokens/s;修改前 20.70 requests/s、22721.25 total tokens/s、2069.85 output tokens/s(折算约 +15%)。作者另注:DeepSeek R1 因卡不足未实测,"guessing it could improve like 15%+"。精度(gsm8k 5-shot,Qwen3-30B-A3B-FP8,enforce_eager,max_model_len=32768):开 DeepGEMM 0.8476/0.8855(flexible/strict),不开 0.8347/0.8939。

## 验证与回归风险

依赖 tests/kernels/moe/test_deepgemm.py 与 tests/kernels/quantization/test_block_fp8.py(统一入口后验证 DeepGEMM 数值与 block-fp8 GEMM 正确性)。回归风险集中在:Blackwell + DeepGEMM 组合下 UE8M0 scale 路径(use_ue8m0 生效时 scale 被向上取整,数值行为改变)、column-major scales 布局,以及公开 wrapper 被删除可能影响外部调用方。

## 关联

前序:#20087。同分类(Quantization)相关卡:#29929(MoE kernel 选择)、#63915(MXFP8 融合量化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
