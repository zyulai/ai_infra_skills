# PR #67393: [Perf][Qwen3.5] Fuse dynamic per-token FP8 input quant into the Gemma add-RMSNorm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LantaoJin |
| 合并 | 2026-11-16 |
| 分类 | Quantization(tier: major) |
| Label | ready, ci/build, qwen, nvidia, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67393 |

## 一句话结论

把 Qwen3.5 NVIDIA 路径的 fused Gemma add-RMSNorm + input quant(#64509)扩展到动态 per-token FP8(llm-compressor `FP8_DYNAMIC` checkpoint 与 online PTPC FP8 的激活方案),norm+quant 在 decode 尺寸(M=1-64)快 1.6-1.9x;64 层模型每个 decode 步消掉 128 个独立 quant kernel;SM120 服务 TPOT -1.2%(c1) / -1.7%(c8)。

## 动机

FP8-dynamic / online PTPC FP8 checkpoint(如 `RedHatAI/Qwen3.5-*-FP8-dynamic`)此前在每个 `input_layernorm` 与 `post_attention_layernorm` 之后还要单独跑一次 `dynamic_per_token_scaled_fp8_quant`;静态 FP8 与 NVFP4 已有 fused 路径(#64509),per-token FP8 缺席。TP1 或 FlashInfer AR+norm 不覆盖的场景(如 PCIe-only GPU)下,每个 decoder 层两个 norm 各多一个 kernel。

## 优化点

- Triton kernel `_gemma_add_rms_norm_quant_kernel` 增加 `QUANT = "fp8_token"` epilogue:对 bf16 归一化后的激活算行 amax,用 `max(amax / 448, 1 / (448 * 512))` 的 IEEE 除法量化(`FP8_MIN_SCALE = 1.0 / (448.0 * 512.0)`,与 csrc `min_scaling_factor` 一致),写出 `(M, 1)` fp32 token scales——与 `dynamic_per_token_scaled_fp8_quant` 数值契约一致。
- `>384` 行时 per-token 用 8 warps 而非静态路径的 4(行 amax 需要;SM120 上扫过)。
- `CutlassFP8ScaledMMLinearKernel.input_quant_key()` 同时暴露 `kFp8DynamicTokenSym`(kernel 本来就接受 unpadded `(M, 1)` token scales);其他 CUDA producer 未注册该 key,故只有此路径改变。
- FlashInfer 无 per-token AR+norm+quant pattern:在其 AR+norm 生效处(NVLink)`forward_fused` 保持 AR+norm、linear 照旧量化;否则(PCIe-only、超 workspace budget)显式 all-reduce 后跑 fused norm-quant(`fused_allreduce_gemma_rms_norm_quant` 中 `kARResidualRMSNorm` pattern + 事后 `ops.scaled_fp8_quant`)。
- GDN gated norm 与 attention sigmoid gate 保持 static-only fusion(一个 program 拥有一个 head 而非一行,per-token amax 需跨 program 归约)。

## 关键改动

```python
# vllm/models/qwen3_5/nvidia/ops/fused_norm_quant.py
# min_scaling_factor in csrc/quantization/utils.cuh: 1 / (448 * 512).
FP8_MIN_SCALE = tl.constexpr(1.0 / (448.0 * 512.0))
```

kernel epilogue(pack diff 摘录):

```python
    elif QUANT == "fp8_token":
        # dynamic_per_token_scaled_fp8_quant numerics: IEEE divisions by
        # max(amax / 448, min scale). Masked columns are 0, so amax is exact.
        y32 = y.to(tl.float32)
        token_scale = tl.maximum(
            tl.math.div_rn(tl.max(tl.abs(y32)), FP8_MAX), FP8_MIN_SCALE
        )
        q = tl.clamp(tl.math.div_rn(y32, token_scale), -FP8_MAX, FP8_MAX)
        tl.store(q_ptr + row * H + cols, q, mask=mask)
        tl.store(sf_ptr + row, token_scale)
```

```diff
 # vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
-        """Only static per-tensor activation quantization is supported for external
-        quantization."""
-        if self.config.activation_quant_key == kFp8StaticTensorSym:
-            return kFp8StaticTensorSym
+        key = self.config.activation_quant_key
+        if key in (kFp8StaticTensorSym, kFp8DynamicTokenSym):
+            return key
```

## 性能声明

- Kernel 微基准(H=5120,CUDA graph of 20 calls,us/call):M=1 4.51 → 2.77(1.63x)、M=3 1.71x、M=8 5.42 → 2.87(1.89x)、M=64 1.87x、M=256 1.28x、M=512 14.10 → 11.11(1.27x)、M=2048 1.30x、M=8192 617.60 → 396.79(1.56x)。
- 端到端:Qwen3.6-27B FP8-dynamic(per-channel 权重、per-token 激活),8x RTX PRO 4500 Blackwell(SM120,PCIe,SageMaker `ml.g7.48xlarge`),TP4 x DP2、MTP-2、FlashInfer attention、prefix caching;TPOT c1 7.156 [7.085, 7.226] → 7.070 [7.011, 7.130] ms(-1.2%),c8 8.146 [7.949, 8.378] → 8.007 [7.827, 8.201] ms(-1.7%)(40 串行 / 5x8 并发请求、固定约 2k prompt、greedy、512 输出 token、bootstrap 95% CI;在 >384-row warps 改动之前测得)。
- 精度:GSM8K(1319, greedy, 0-shot chat)0.962 → 0.965。
- 正确性契约:融合 kernel 与 unfused 链(`gemma_fused_add_rmsnorm` + `scaled_fp8_quant(..., use_per_token_if_dynamic=True)`)产生相同 FP8 codes 与 `(M, 1)` token scales——normed bf16 值匹配的行逐位相同,1-ulp 差异的行至多差一个 code(与静态 FP8 路径同一契约)。

## 验证与回归风险

- 单测(SM120):`pytest tests/models/qwen3_5/test_gemma_add_rms_norm_quant.py tests/models/qwen3_5/test_fused_allreduce_norm_quant.py` 本 PR 105 passed;base(新测试对旧代码)25 failed / 80 passed,证明 24 个新 `fp8_token` kernel case 先失败后通过。
- `test_forward_fused` 的 bitwise 断言改为以 `flashinfer_allreduce_gemma_rms_norm_applies` 为准(SM120 上 FlashInfer AR+norm 不跑,main 上该测试因 1.7M 个 normed 值里 19 个差 1 ulp 而失败;B200 不变)。
- 风险:SM90 未验证(CUTLASS per-token 路径现在也走 fused kernel);CI 在 "(B200) Qwen3.5" 步运行两个测试文件,其 `source_file_dependencies` 新增 `cutlass.py` 与 `fused_allreduce_gemma_rms_norm.py`。
- 重复实现检查:#65497/#45770(torch.compile RMSNorm+quant pass)、#45364/#61073(Llama 式 RMSNorm 手工融合)均不在本路径。

## 关联

- 扩展 #64509 的 fused Gemma add-RMSNorm + input quant 到 per-token FP8;消费方为 llm-compressor `FP8_DYNAMIC` checkpoint 与 online PTPC FP8。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
