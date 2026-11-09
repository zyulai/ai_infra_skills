# PR #33471: runtime: Add flashinfer rmsnorm + quant fusion support SM90, SM100, SM120- #32994

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DevashishLal-CB |
| 合并 | 2026-08-09 |
| 分类 | Quantization | tier: major |
| Label | performance, quant, blackwell, run-ci, nvidia, run-ci-extra, kernel |
| Diff | diffstat 缺失(素材未提供统计数字);diff 含约 12 个文件,另注明有 1 个文件因超长被裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/33471 |

## 一句话结论

把静态 per-tensor FP8 activation 量化融合进 flashinfer RMSNorm kernel,省去独立的 quant kernel 与一次内存往返。

## 动机

标准推理路径中,RMSNorm 输出 bf16 激活后还要由下游 FP8 linear 各自做一次 static per-tensor activation quant,形成两个 kernel 和多余的显存读写。这是 #32994 的 runtime 部分,目标是在 SM90/SM100/SM120(GPU,含 Blackwell)上利用 flashinfer 的 `rmsnorm_quant` / `fused_add_rmsnorm_quant` kernel,把 norm 与 FP8 量化合并为一次计算,并让下游 FP8 GEMM 直接消费预量化激活。

## 优化点

机制分三段:(1) `RMSNorm.forward_cuda` 新增 `quant_linear` 参数,当下游 linear 是"静态 per-tensor FP8 activation scale"的 FP8 linear(识别原生 `Fp8LinearMethod` 非 block/mxfp8/marlin,以及 compressed-tensors W8A8-FP8 static 输入方案)时,走新增的 `forward_with_per_tensor_quant_fusion`,调用 flashinfer fused kernel 一步产出 `(fp8_out, scale, orig_dtype)`;有 residual 时用 `fused_add_rmsnorm_quant` 就地更新 residual。(2) `Fp8LinearMethod.apply` 与 compressed-tensors `apply_weights` 支持接收 tuple 形式的预量化激活,跳过重新量化,并携带 `pre_quant_output_dtype` 保证 GEMM 输出 dtype 正确。(3) `apply_fp8_linear` 重构 scale 分发:预量化输入直接使用传入 per-tensor scale;并新增 SM90/SM120 支持判断,CUTLASS channelwise GEMM 在这些架构上可用标量 A scale,避免逐行 repeat scale 的额外开销。llama/qwen2 及其 eagle 模型在调用 layernorm 处传入 `quant_linear=qkv_proj`/`gate_up_proj` 接通该路径。

## 关键改动

```text
python/sglang/srt/layers/layernorm.py: 新增融合分发与 forward_with_per_tensor_quant_fusion
python/sglang/srt/layers/quantization/fp8_utils.py: apply_fp8_linear 支持预量化 fp8 输入与 SM90/SM100/SM120 标量 A scale
python/sglang/srt/layers/quantization/fp8.py: Fp8LinearMethod.apply 接收预量化 tuple 输入
python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py: apply_weights 同上
python/sglang/srt/models/llama.py, qwen2.py: layernorm 调用传入 quant_linear 接通融合路径
benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py: 新增 fused vs unfused 微基准
```

关键片段(forward_cuda 的融合门控):

```python
if (
    quant_linear is not None
    and not self.cast_x_before_out_mul
    and _flashinfer_rmsnorm_quant_available
):
    scale = _fp8_static_input_scale(quant_linear)
    if scale is not None:
        return self.forward_with_per_tensor_quant_fusion(
            x, scale, residual, post_residual_addition
        )
```

## 性能声明

PR 正文未提供 benchmark 数字,仅说明为 #32994 的 runtime 改动。素材中新增的微基准脚本 `benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py` 用于对比 unfused(RMSNorm + static_quant_fp8)、flashinfer 默认 fused 与 CuTe-DSL fused 三种实现(hidden_size 4096/8192、带/不带 residual、num_tokens 512-16384),但 PR 正文未附测试结果,PR 未注明测试条件。

## 验证与回归风险

新增注册 CI 测试 `test/registered/layers/test_layernorm_fusion.py`(TestRMSNormFp8QuantFusion):覆盖 bf16/half、多个 num_tokens/hidden_size、有无 residual,校验 fp8 输出契约与 cos>0.99、rel_err<0.1;并测试 dispatch 门控(variance_size_override 与 cast_x_before_out_mul 场景不得走融合)。`test/registered/quant/test_fp8_utils.py` 新增 TestApplyFp8LinearScaleDispatch 校验 SM90/SM100/SM120 下的 scale 分发。风险点:改动 `apply_fp8_linear` 是所有 FP8 linear 的公共路径,非静态 per-tensor 量化(block-quant、per-token dynamic、marlin、mxfp8)需保证不误入预量化分支;llama_eagle/qwen2_eagle 的 layer-0 identity layernorm 签名同步修改,其他以 lambda 替换 layernorm 的模型若有遗漏会报参数错误。

## 关联

#32994(本 PR 为其 runtime 配套部分)。同分类区域:FP8 量化与 norm 融合相关改动,如 #13617(ROCm DeepSeek R1 FP8)、#12307(trtllm fp8 attention)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
