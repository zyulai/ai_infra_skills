# PR #58800: [ROCm] gfx1250 optimizations and model enablement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jpvillam-amd |
| 合并 | 2026-10-15 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | rocm, ready, ci/build, deepseek, gpt-oss, quantization, DSv4, minimax |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「8 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/58800 |

## 一句话结论

为 gfx1250 启用 gpt-oss-120b、DeepSeek-R1-0528、DeepSeek-V4-Flash、MiniMax-M3（MXFP4）的优化 fast path。

## 动机

PR 为 gfx1250（ROCm）启用并优化 gpt-oss-120b、DeepSeek-R1-0528、DeepSeek-V4-Flash 与 MiniMax-M3（MXFP4）。正文强调所有改动仅限 gfx1250，其他 ROCm 与 CUDA 路径行为不变。正文并称这不是重复 PR：没有其他 open PR 针对这些模型的 gfx1250 enablement，Kimi-K3 的 gfx1250 支持被拆分出去单独 upstream。

## 优化点

MoE：新增 AITER Triton/Gluon W4A4 MXFP4 专家 `aiter_mxfp4_w4a4_moe.py`，配独立 backend enum，使 oracle 仅在 gfx1250 选中它（MI300/MI355 默认不变）；另有 gfx1250 MX-scale swizzle、W4A16/W4A8 权重准备、router-logit sanitization、fp32 routing bias 与 routing-replay plumbing。Linear：存在 tuned config 时使用 AITER preshuffled block-FP8 GEMM。Attention / sampling：AITER unified-attention 后端新增 gfx1250 路径，并加入 AITER fused sampler（`sampler.py`、`topk_topp_sampler.py`）。Models：DeepSeek-V4 AMD 修复与 gpt-oss MXFP4 stride 修复；gpt-oss RMSNorm+pad 融合复用既有 `fuse_act_padding` pass，而非 model-specific op。Docker：更新 gfx1250 base/image（ROCm SDK、Triton 与 AITER pins）。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a4_moe.py: 新增 373 行 AITER Triton/Gluon MXFP4 W4A4 专家（AiterW4A4ExpertsMonolithic、aiter_triton_kernel_w4a4_moe_forward）。
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py、vllm/config/kernel.py: 新增 AITER_TRITON_MXFP4_MXFP4 backend enum 与 "aiter_triton_mxfp4_mxfp4" 配置项，oracle 仅在 gfx1250 映射到该 W4A4 专家。
vllm/_aiter_ops.py: 注册 rocm_aiter_triton_gemm_a8w8_blockscale_preshuffle custom op，并新增 is_triton_blockscale_bpreshuffle_gemm_tuned（AITER preshuffled block-FP8 GEMM 的 tuned 探测）。
tests/kernels/moe/test_ocp_mx_moe.py: 新增 ROCM_GFX1250 / on_gfx1250() 探测与 AITER_TRITON_MXFP4_MXFP4（requires_gfx1250）测试项。
vllm/model_executor/layers/quantization/（mxfp4.py、quark/quark_moe.py、utils/mxfp4_utils.py）: gfx1250 MX-scale swizzle 与权重准备相关改动。
```

```text
    # W4A16: aiter Triton moe_gemm_a16w4 kernel (gfx942/gfx950/gfx1250)
    AITER_TRITON_MXFP4_BF16 = "AITER_TRITON_MXFP4_BF16"
    AITER_MXFP4_FP8 = "AITER_MXFP4_FP8"  # W4A8: triton kernel
    AITER_MXFP4_MXFP4 = "AITER_MXFP4_MXFP4"  # W4A4: CK kernel (gfx950)
    # W4A4: aiter Triton/Gluon moe_gemm_a4w4 kernel (gfx1250)
    AITER_TRITON_MXFP4_MXFP4 = "AITER_TRITON_MXFP4_MXFP4"
```

## 性能声明

PR 正文未给出定量性能数据。Test Plan 为 "gsm8k (lm-eval) for all models along side the normal server benchmarks"；Test Result 仅给出 PR 评论链接（https://github.com/vllm-project/vllm/pull/58800#issuecomment-5902957137），正文无 benchmark 数字、模型/硬件/精度/workload 条件，PR 未注明测试条件。

## 验证与回归风险

验证：正文 Test Plan 为 gsm8k（lm-eval）覆盖所有模型加常规 server benchmarks；diff 中 `tests/kernels/moe/test_ocp_mx_moe.py` 新增 gfx1250 分支（`ROCM_GFX1250`、`on_gfx1250()`、`requires_gfx1250` 跳过逻辑与 `AITER_TRITON_MXFP4_MXFP4` 测试项）。回归风险：正文强调所有改动仅限 gfx1250，其他 ROCm 与 CUDA 路径行为不变，MI300/MI355 默认不变；涉及 gpt-oss、DeepSeek-V4、MiniMax-M3 多模型与 MoE / Linear / Attention / sampling / 量化多条路径，且 pack 注明 8 file(s) dropped from diff（完整 diff 需看 PR）。

## 关联

无（正文称无其他 open PR 针对这些模型的 gfx1250 enablement；Kimi-K3 gfx1250 支持被拆分单独 upstream，未给出 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
