# PR #40710: [Aiter][ROCm] RMSNormGated+GroupedQuantFP8 fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tpopp |
| 合并 | 2026-05-14 |
| 分类 | Quantization (tier: major) |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 6 个文件(fusion pass、matcher、aiter ops、layernorm 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/40710 |

## 一句话结论

把 RMSNormGated+reshape+group FP8 量化融成单个 AITER kernel,Qwen3-Next 吞吐约 +2.3%。

## 动机

PR 正文 Motivation:在 GatedDeltaNetAttention 模型(如 Qwen3-Next-80B-A3B-Instruct-FP8)里,每个 attention 输出路径依次做 RMSNormGated(对 per-head 张量 (N*H, D) 加 gating)、reshape 回 (N, H*D)、再按 GroupShape(1,128) 做 group FP8 量化,然后送输出投影 linear。torch.compile 会把这三步降解成大量 elementwise 与 reduction kernel,带来多次 GPU kernel launch 与中间显存流量。

## 优化点

新增编译融合 pass `AiterRMSNormGatedFp8GroupQuantPattern`,在 FX graph 中匹配「分解后的 norm → reshape → quant」子图,替换为单个 AITER Triton kernel `fused_rms_gated_fp8_group_quant`(通过注册 `rocm_aiter_fused_rms_gated_fp8_group_quant` custom op 包装)。为让模式匹配,新增 `_fold_consecutive_reshapes` 预处理(make_fx 会忠实记录链式 reshape,需先折叠),并经 static_forward_context 动态推断 num_heads/head_dim。`matcher_utils.py` 增加 `MatcherRMSNormGated` tracer 并扩展 `MatcherQuantFP8` 支持 Triton 量化 op;`layernorm.py` 把 `RMSNormGated.forward_static` 抽成 @staticmethod 供 forward_native 与 matcher 共用。pass 由 `are_gdn_triton_kernels_available()` 门控,旧 AITER 上优雅 no-op。

## 关键改动

```text
vllm/compilation/passes/fusion/rocm_aiter_fusion.py: 新增 AiterRMSNormGatedFp8GroupQuantPattern 与 _fold_consecutive_reshapes,动态推断 num_heads/head_dim
vllm/_aiter_ops.py: 注册 fused_rms_gated_fp8_group_quant custom op 与 are_gdn_triton_kernels_available() 检查
vllm/compilation/passes/fusion/matcher_utils.py: 新增 MatcherRMSNormGated、扩展 MatcherQuantFP8 支持 Triton quant
vllm/compilation/passes/vllm_inductor_pass.py: 提供 reshape 折叠工具
vllm/model_executor/layers/layernorm.py: 抽出 RMSNormGated.forward_static 静态方法
tests/compile/passes/test_fusion.py: 新增 TestGatedModel 正/负匹配用例
```

关键机制代码(模式定义):

```python
class AiterRMSNormGatedFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    Matches decomposed RMSNormGated + reshape + group FP8 quant and replaces
    with rocm_aiter_fused_rms_gated_fp8_group_quant.
    """
    FUSED_OP = rocm_aiter_ops.get_fused_rms_gated_fp8_group_quant_op()
```

## 性能声明

PR 正文 Results:两个共 9us 的 kernel 可合并为 4.5us,Qwen3Next 上视 workload 大小(concurrency 1 vs 128)有 1-3% improvement。Setup:Qwen/Qwen3-Next-80B-A3B-Instruct-FP8、TP=1、AMD MI355x(gfx950)单卡、vllm/vllm-openai-rocm:nightly(v0.19.2rc1)+ AITER main + ROCm/aiter#2423、backend ROCM_AITER_FA、cudagraph_mode=FULL_AND_PIECEWISE、pass_config={"fuse_norm_quant": true};命令 `vllm bench serve --dataset_name random --random_input_len 1024 --random_output_len 1024 --max_concurrency 4 --num_prompts 32 --num_warmups 4 --seed 1 --temperature 0 --ignore_eos`。Throughput(ISL=1024/OSL=1024/c=4):output token throughput 456.52 → 467.05 tok/s(+2.3%)、total 913.04 → 934.11(+2.3%)、mean TPOT 8.66 → 8.44 ms(−2.5%)、P99 TPOT 8.98 → 8.67(−3.5%)、mean E2EL 8971 → 8769 ms(−2.3%)。精度 lm_eval gsm8k 5-shot:flexible-extract 0.8506 → 0.8605、strict-match 0.8097 → 0.8089,均在 error bars 内。

## 验证与回归风险

Test plan:`pytest tests/compile/passes/test_fusion.py -k "gated"` 正/负 pattern match 单测、lm_eval gsm8k 精度、vllm bench serve 吞吐,并验证 AITER 缺 GDN kernel 时 graceful no-op。pattern 匹配验证:融合开启时替换 5 个 pattern(其中 4 个来自本 PR),注释掉后仅替换 1 个。影响路径:ROCm + AITER GDN triton kernel 可用时的 GatedDeltaNetAttention(Qwen3-Next 等)输出路径;依赖外部 ROCm/aiter#2423 提供 fused kernel,旧 AITER 上自动回退。

## 关联

外部依赖 ROCm/aiter#2423("[Triton] optimized decode kernels for Qwen3-Next model",提供 fused_rms_gated_fp8_group_quant);正文未引用其他 vLLM PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
