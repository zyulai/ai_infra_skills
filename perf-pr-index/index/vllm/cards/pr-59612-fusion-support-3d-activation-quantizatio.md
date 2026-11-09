# PR #59612: [Fusion] Support 3D activation quantization fusion and enable it for Pixtral

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jzakrzew |
| 合并 | 2026-10-09 |
| 分类 | Quantization(major) |
| Label | ready, nvidia, quantization, verified, mistral |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 fusion/kernels/model/测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/59612 |

## 一句话结论

把 SiLU+multiply+quant fusion 生产者从 2D 扩展到 3D 输入并接通 NVFP4 consumer,使 Pixtral vision MLP 直接走 maybe_fused_act_quant,RTX6000D 上 NVFP4 prefill +1.9%。

## 动机

Pixtral vision MLP 的输入是 [batch, tokens, hidden] 形状,把 gate/up 投影输出直接传给既有 dynamic block-FP8/NVFP4 生产者会触发其 2D-only 断言,vision encoder 无法使用 activation+quant fusion,只能走普通 activation 路径。shape 处理放在共享 producer 层,调用方即可不经模型侧 reshape 使用 fusion helper。

## 优化点

fused_act_quant.py 中 _silu_and_mul_fp8_dynamic_block 与 _silu_and_mul_nvfp4_dynamic 从 2D-only 放宽到 2D/3D:先保存 activation 输出 shape,把 token 维展平后调用既有 2D 内核,再恢复 data 与 orig_shape 元数据,2D 行为不变;static-FP8 producer 本就支持 leading dim,无需改。FlashInfer CuTe-DSL NVFP4 consumer 新增 input_quant_key()=kNvfp4Dynamic 的 pre-quantized 输入支持;FlashInfer-CUTLASS consumer 在 GEMM 前把 3D packed data 展平、输出后恢复 token 维。两个 Pixtral MLP forward(原生与 HF 版)改为在 gate/up 与 down 投影之间调用 maybe_fused_act_quant;不支持的 activation/quant 组合与 consumer 保持原路径。

## 关键改动

```text
vllm/model_executor/layers/fusion/fused_act_quant.py: 两个 producer 接受 2D/3D 输入
vllm/model_executor/kernels/linear/nvfp4/flashinfer.py: CuTe-DSL 接受 QuantizedActivation,
  CUTLASS 展平/恢复 3D packed data
vllm/model_executor/models/pixtral.py: 两个 MLP forward 接入 maybe_fused_act_quant
```

## 性能声明

Ministral 3B prefill throughput,RTX6000D,请求含 1700×2200 像素扫描 US Letter 页图像(vidore v3),NVFP4 vision encoder:6.521 → 6.642 req/s,约 +1.9%(相对 #57168 后的 compiled baseline);BF16 路径差异约 0.2%。同表还给出同 revision 下 NVFP4 vs BF16 为 +23.0%。

## 验证与回归风险

扩展现有 static-FP8 与 dynamic block-FP8 fusion 测试至 3D 输入(数值 parity、输出 shape、quantization metadata);新增 FlashInfer-CUTLASS NVFP4 consumer 回归测试(flat 与 3D QuantizedActivation 输入结果一致、dtype 与 token 维恢复)。命令:tests/kernels/test_fused_quant_activation.py、tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py -k preserves_quantized_input_shape。风险集中在 NVFP4/FP8 fusion 对 3D 输入的处理与 consumer 的形状恢复;不支持的组合回落原路径。

## 关联

#43501(相关);基线包含 #57168(merged)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
