# PR #39920: [Moe] Fix flashinfer_trtllm silently dropping swiglu_limit clamped SwiGLU activation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39920](https://github.com/sgl-project/sglang/pull/39920) |
| 作者 | Dovis01 |
| 合并日期 | 2026-09-17 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 3 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
flashinfer_trtllm FP8 MoE 路径只物化 gemm1_clamp_limit,用 swiglu_limit 传 clamp 的模型(如 GLM-5.3-Flash)被静默降级为无 clamp SwiGLU;修复为 gemm1_clamp_limit 缺省时回退到 swiglu_limit。

## 问题现象
带 clamped SwiGLU 激活的模型(如 GLM-5.3-Flash / GLM-5-Next、Qwen3-Next 风格)通过 `FusedMoE(swiglu_limit=...)` -> `MoeRunnerConfig.swiglu_limit` 传递 clamp 上限。所有 MoE runner 后端都应遵守:triton 把它传入融合 kernel,deep_gemm 通过 `silu_and_mul_clamp` / `_apply_swiglu_limit` 应用,但 `flashinfer_trtllm` 静默忽略它。无报错、无警告,专家激活从训练时的 `silu(clamp(gate, max=L)) * clamp(up, -L, L)`(L=10.0)退化为普通 `silu(gate) * up`,可测地损害精度。GLM-5.3-Flash FP8 上 GSM8K 5-shot 对比:修复前 flashinfer_trtllm 0.9265/0.9280/0.9287(均值 0.9277),修复后 0.9348/0.9371(均值 0.9360),与 triton 基线(0.9360)完全持平,前后分布不重叠(组内方差约 0.2pp)。正文标注 fix issue #39797。

## 根因
正文披露:flashinfer_trtllm FP8 路径只物化 `gemm1_clamp_limit`(TRT-LLM kernel 参数),而对使用 `swiglu_limit` 字段的模型该值保持 `None`。两条命名管线(`gemm1_clamp_limit` 由 gpt_oss / bailing_moe_v3 使用,`swiglu_limit` 由 DeepseekV2MoE 系模型使用)在 `_prepare_flashinfer_trtllm_activation_params` 汇合,该函数只取前者,于是后者被静默丢弃。

## 修复方案
单文件改动:`python/sglang/srt/layers/quantization/fp8.py` 的 `Fp8MoEMethod._prepare_flashinfer_trtllm_activation_params()`,当 `gemm1_clamp_limit` 未显式设置时回退到 `swiglu_limit`:

```python
+        clamp_limit = (
+            self.moe_runner_config.gemm1_clamp_limit
+            if self.moe_runner_config.gemm1_clamp_limit is not None
+            else self.moe_runner_config.swiglu_limit
+        )
         for name, value in (
             ("gemm1_alpha", self.moe_runner_config.gemm1_alpha),
             ("gemm1_beta", self.moe_runner_config.gemm1_beta),
-            ("gemm1_clamp_limit", self.moe_runner_config.gemm1_clamp_limit),
+            ("gemm1_clamp_limit", clamp_limit),
```

正文说明放这里而非模型代码的原因:该函数是构建 TRT-LLM SwiGLU 参数(gemm1_alpha/beta/clamp_limit)的唯一消费点;triton 独立消费两个字段,在模型层同时设置会改变无关后端行为,而此回退只作用于 flashinfer_trtllm 准备阶段。TRT-LLM kernel 原生支持 clamp:`ActivationType.Swiglu` 下 `gemm1_clamp_limit=L` 计算 `X2 = clamp(X2, max=L)`、`X1 = clamp(X1, -L, L)`、`out = X2 * sigmoid(alpha * X2) * (X1 + beta)`(alpha 默认 1.0、beta 默认 0.0),与 `models/glm5_next.py` 的参考 `swiglu_clamped` 完全一致;已在 flashinfer 0.6.18(`trtllm_fp8_block_scale_moe`、`Fp8QuantizationType.DeepSeekFp8`)上验证。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 flashinfer_trtllm FP8 MoE 路径的激活参数物化:已显式设置 `gemm1_clamp_limit` 的模型(gpt_oss、bailing_moe_v3)行为不变;其余经 `swiglu_limit` 传 clamp 的模型(如 Qwen3-Next)在 flashinfer_trtllm 下激活行为改变(由无 clamp 变为有 clamp)。开销方面只在权重加载时多物化一个每专家 fp32 tensor(此处 288 字节),热路径 kernel 调用不变,正文称无吞吐回退。

## 关联
- [#39797](https://github.com/sgl-project/sglang/issues/39797)(本修复对应的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
