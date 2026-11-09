# PR #62057: [ROCm][Model] Add Beam 1.0 support on ROCm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jamesETsmith |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, new-model, rocm, frontend, speculative-decoding, ready, ci/build, tool-calling, qwen, nvidia, mrv2, dflash, rust, beam, recipe |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);正文注明 1,968 additions / 2 deletions、12 个 Beam model 文件;diff 另有 8 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/62057 |

## 一句话结论

把 Reflection Beam 1.0(501B/23B-active MoE)与 DFlash2 drafter 移植到 ROCm(AMD-owned 实现、平台分发):8x MI350X 上 DFlash2 投机解码带来 1.33x-1.57x 吞吐提升、TTFT 显著下降。

## 动机

#61978 为 NVIDIA 提供 Beam 1.0 的 day-0 支持,但其 kernels(含 EP 用的 custom all-reduce)为 CUDA-only,recipe 将 AMD 全系标为 unsupported。本 PR 是 ROCm follow-up:在 `vllm/models/beam/amd/` 落地 AMD 实现,使 Beam 1.0 BF16 可在 8x MI300X-class 节点(TP8 + expert parallelism)上服务,并支持 DFlash2 投机解码。#61978 已拆分为 #62042、#62052、#62074,本 PR 针对更新后的 main。

## 优化点

`vllm.models.beam` 包按平台分发:ROCm 导出 AMD 类,其他平台导出 NVIDIA 类,相对 diff 不改 `vllm/models/beam/nvidia/`。AMD target 走标准 vLLM ROCm 路径——backend 自有的 K/V cache 更新、标准线性层、finalized MoE 输出、RCCL collectives——替代 NVIDIA 的 direct-cache、CuTe/FlashInfer、MNNVL fast path;Beam 必需的 FP32 激活、归一化、Q/K 预处理、attention-output gate 与 logit softcap 算术保留在 AMD-local Triton kernels 中(`beam_silu_mul`、`beam_softcap_logits`、`beam_qk_norm_rope`、`beam_output_gate`、`beam_rms_norm`、`beam_norm_residual_norm`),以 `enable_fp_fusion=False` 与显式 dtype 舍入边界保住数值契约,并省略 CUDA-only Triton launch 选项(`enable_reflect_ftz`、`launch_pdl`)。DFlash2(1.93B drafter)的 context K 归一化与 RoPE 由 AMD-local Triton 计算,cache 布局、slot scatter 与量化委托给所选 ROCm attention backend 的 native cache-update API(`fuse_kv_insert` 条件:backend forward 不含 KV 更新且支持 bypass、DCP=1、auto/bfloat16 KV dtype)。无 HIP/C++ kernel,不改 scheduler、KV manager、model runner、attention backends 与 NVIDIA 实现。

## 关键改动

```text
vllm/models/beam/__init__.py: 按 ROCm/NVIDIA 平台分发 target 与 drafter 类,统一导出 beam_* 算子
vllm/models/beam/amd/{model,attention,attention_ops,activation,layernorm,dflash2_conv_ops,dflash2}.py: AMD target 与 drafter 实现
vllm/models/beam/amd/attention.py: BeamAttention GQA + fuse_kv_insert 判定(backend cache-update bypass)
recipes/models/reflection/Beam-1.0-501B-A23B.yaml: AMD 硬件支持、hardware_overrides(ROCM_AITER_UNIFIED_ATTN + triton MoE + VLLM_ROCM_USE_AITER)
tests/models/test_beam_{construction,numerics,weight_mapping}.py, tests/v1/spec_decode/test_dflash2.py: 平台化(is_cuda_alike)与 AMD numerics 测试
```

## 性能声明

硬件 8x MI350X(gfx950),ROCm 10.1,PyTorch 2.13,TP8 + expert parallelism,Beam 1.0 BF16 + BF16 DFlash2 drafter(`VLLM_ROCM_USE_AITER=1`,`--attention-backend ROCM_AITER_UNIFIED_ATTN --moe-backend triton`)。NVIDIA SPEED-Bench throughput_8k / low_entropy(固定 1,024-token 输出、greedy、每点 16 warmups + 10×concurrency 个 measured 请求),DFlash2 vs target-only 吞吐:c=1 117.40→168.47 tok/s(1.44x)、c=4 331.88→443.00(1.33x)、c=8 492.26→769.78(1.56x)、c=16 803.81→1,263.22(1.57x);TTFT 全点更低(如 c=8 634.83→240.43 ms);acceptance length 3.41-3.63,token acceptance 30.13%-32.83%;580 个请求全部成功,单次运行无变异性估计。精度:GSM8K 1,319 题(5-shot chat、greedy、TP8+EP)strict extraction:target 1,202/1,319(91.13%±0.78%)、DFlash2 1,200/1,319(90.98%±0.79%);flexible extraction 两者均 1,212(91.89%);MI355X 上 vanilla 与 DFlash2 均 90.06%(GPQA-Diamond)、97.45%(AIME 2026);MI325X 均 90.18%(GPQA-Diamond),与 Reflection 公布的 BF16 精度在统计误差内一致。

## 验证与回归风险

`tests/models/test_beam_numerics.py` 覆盖 AMD Triton kernels 的数值边界(softcap 舍入、SiLU 乘积舍入、FP32 affine、residual 中间 BF16 舍入、QK norm+RoPE、cache 插入 slot mapping 与 replay、跨 token 数复用 specialization——unexpected-compile fail 扩展到 `vllm.models.beam.amd`);`test_beam_weight_mapping.py`、`test_dflash2.py` 关键子集进入 buildkite;NVIDIA-specific 测试(beam multicast all-reduce、packed projection)在 ROCm skip。回归风险:AMD 模型路径在 sequence-parallel MoE(DEP)下直接 raise(recipe 将 `single_node_dep` 标为 amd unsupported);KV insert fusion 仅限 auto/bfloat16 KV cache;MI300X 走未实测的 gfx942 代码路径由 MI325X 验证覆盖。

## 关联

#61978(NVIDIA day-0 支持,拆分为 #62042、#62052、#62074);recipes#1107(recipe 源);同分类 Speculative Decoding 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
