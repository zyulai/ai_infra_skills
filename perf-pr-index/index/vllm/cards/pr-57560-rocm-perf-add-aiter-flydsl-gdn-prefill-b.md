# PR #57560: [ROCm][Perf] Add AITER FlyDSL GDN prefill backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd(co-authored-by @nholmber) |
| 合并 | 2026-10-07 |
| 分类 | Attention / Kernel(tier: major) |
| Label | rocm, ready |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(6 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/57560 |

## 一句话结论

在 ROCm 上新增 opt-in 的 AITER FlyDSL GDN(linear attention)prefill backend:MI355X 上 Qwen3.8-2.4T-A95B 的 16 组 serving A/B 中 15 组 TTFT 改善,geomean -3.6%。

## 动机

Qwen3.8 等 GDN 模型在 ROCm 上的 prefill 此前只能走 Triton/FLA 路径;AITER 上游新增了优化的 VK prefill 实现(fused FlyDSL prepare stage + FlyDSL K5 chunk kernel,K5 按 64-token 固定 chunk 编译),需要接入点让受支持的平台/形状/dtype 用上这些 kernel,不支持的组合保留 Triton/FLA 路径。依赖 ROCm/aiter#4532、#4598、#4694、#4732 与 prepare-kernel 修复 #4952(均已合并)。

## 优化点

`--gdn-prefill-backend` 新增 `aiter_flydsl` 选项(纯 opt-in)。`_resolve_gdn_prefill_backend` 的 ROCm 分支:要求 `VLLM_ROCM_USE_AITER=1` 且 AITER FlyDSL prefill kernels 可导入(`chunk_gated_delta_rule_fwd_h_flydsl_opt`、`gdn_prepare_fwd_flydsl` 等),模型为 BF16 且 `head_k_dim == head_v_dim == 128`;形状/dtype 不满足则告警并回退 Triton,kernel 不可导入或非 ROCm 平台显式请求则直接抛错(fail closed,避免静默跑 Triton 却把 run 报告为 FlyDSL)。执行路径 `rocm_aiter_ops.gdn_flydsl_prefill` 调 AITER `chunk_gated_delta_rule_opt_vk`(`use_chunk_flydsl=True, use_prepare_flydsl=True`,chunk size 64),参数逐个具名传递以防 AITER 端改名静默错绑;metadata builder 用 AITER 自带的可复用 varlen prefill metadata 取代 FLA chunk metadata,`GDNAttentionMetadataBuilder._check_chunk_metadata_override` 会拒绝 override 了 `_build_chunk_metadata` 的子类(Kimi K3 KDA 类模型由 resolver 提前拒绝)。

## 关键改动

```text
vllm/_aiter_ops.py: FlyDSL kernel 可导入性检查、可复用 varlen metadata 构建、gdn_flydsl_prefill 封装(GDN_FLYDSL_CHUNK_SIZE=64)
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: backend 解析门控(BF16 + 128/128)、forward_aiter_flydsl
vllm/v1/attention/backends/gdn_attn.py: metadata 增加 aiter_prefill_metadata;chunk metadata override 防护
vllm/engine/arg_utils.py: --gdn-prefill-backend 增加 aiter_flydsl
tests/v1/attention/test_gdn_aiter_flydsl.py: 门控与数值一致性测试
```

## 性能声明

硬件 8x MI355X(gfx950),ROCm 7.2.3;模型 amd/Qwen3.8-2.4T-A95B-Quark-MXFP4,TP8,BF16 GDN,FP8 KV cache,OSL 1024。Serving A/B:16 组 ISL(8k/16k/60k/128k)× 并发(1/2/4/8)单元中 15 组 Med TTFT 改善,幅度 -1.6% 至 -5.6%(仅 8k c=8 为 +1.0%),geomean -3.6%;ITL 全部在 ±0.3% 内(prefill-only 改动,decode 持平)。精度(GPQA-Diamond 与 GSM8K,thinking enabled,checkpoint 默认 FP32 SSM state):GPQA stock 0.9192(182/198)vs FlyDSL 0.9343(185/198);GSM8K stock 0.9787(1286/1314)vs FlyDSL 0.9825(1291/1314)。

## 验证与回归风险

新增 `tests/v1/attention/test_gdn_aiter_flydsl.py`(MI355X 上 9 passed):数值一致性(packed sequences [70,130],与同一 AITER 入口关闭 `use_chunk_flydsl`/`use_prepare_flydsl` 作参考,rtol/atol 2e-2,按 SSM state dtype 参数化 fp32/bf16,并断言 final state dtype)、shape/dtype 门控、KDA 风格模型拒绝、fail-closed。回归影响:仅影响显式请求 `aiter_flydsl` 的 ROCm 部署;其他组合回退 Triton;CUDA 平台解析路径不变(ROCm 分支提前返回);依赖含 5 个上游 PR 的 AITER 版本。

## 关联

依赖 ROCm/aiter#4532、#4598、#4694、#4732、#4952。同分类 Attention / Kernel 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
