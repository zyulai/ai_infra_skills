# PR #30541: [MoE Backend] Add HPC-Ops FP8 MoE runner backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-07-24 |
| 分类 | MoE |
| Label | performance, run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 10 个文件,其中 3 个新文件(hpc_ops.py、数值测试、guard 测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/30541 |

## 一句话结论

新增 opt-in 的 `hpc_ops` MoE runner backend,接入腾讯混元 HPC-Ops 单体 FP8 fused-MoE kernel,小 batch decode 时延最高 4.2x,Hunyuan v3 decode 吞吐 +6%。

## 动机

为 FP8 MoE 模型引入 HPC-Ops(Tencent/hpc-ops,混元 AI Infra 团队的推理算子库)的单体 fused-MoE kernel:一次调用完成 gather → grouped gate_up GEMM → SiLU-and-mul(+FP8 requant)→ grouped down GEMM → weighted top-k reduce,替代现有多段实现。这是 HPC-Ops 与 SGLang 第一期整合的第二部分(attention backend 在 #30540)。kernel 覆盖 sm90+,但目前主要针对 H20 调优,H100/H200/B200 上收益可能有限,因此 backend 严格 opt-in,默认路径不受影响。

## 优化点

新增 `moe_runner/hpc_ops.py`,以 fused-func runner(与 flashinfer_trtllm/cutlass 同形态)提供两条路径:blockwise FP8(128x128 权重块 + 动态 per-token-group-128 激活,如 Qwen3-FP8 / Hy3-FP8 blockwise checkpoint)走 `hpc.fuse_moe_blockwise`;per-tensor FP8(静态激活 scale)走 `hpc.fuse_moe`,per-expert dequant alphas(`w_scale * a_scale`)与激活 requant 的 inverse a2 scale,接线对齐 flashinfer CUTLASS per-tensor 路径。`Fp8MoEMethod` 在 `process_weights_after_loading` 一次性预算 HPC-Ops 要求的 scale 布局(`[E, N/128, K/128]` fp32,K 维 pad 到 4 的倍数),`apply` 时构建 `HpcOpsMoeQuantInfo`。kernel 直接消费全局 top-k expert id 配合 `rank_ep`/`num_expert_total`,故 `hpc_ops` 加入 standard dispatcher 的 `skip_local_expert_mapping`,连续切分 EP 无需 local-expert remap。启动期双重 guard:`MoeRunner` 构造时与 layer 初始化时校验 quant method 必须是 `Fp8MoEMethod`,防止 EP>1 下全局 id 被当 local id 误路由。约束(gated SiLU、无 fused shared experts、无 swiglu clamp、hidden 维 128 对齐、bf16 hidden states)不满足时报清晰错误。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/hpc_ops.py: 新文件,两条 FP8 fused 路径 + quant info + 约束检查
python/sglang/srt/layers/quantization/fp8.py: 预计算 scale 布局,_get_hpc_ops_quant_info
python/sglang/srt/layers/moe/token_dispatcher/standard.py: skip_local_expert_mapping 增加 hpc_ops
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: layer 级 quant method 校验 guard
test/registered/moe/test_hpc_ops_moe.py: 新文件,对 triton 与 fp32-exact 的三方数值测试(未装 HPC-Ops 或 pre-sm90 时跳过)
```

关键机制(scale 预计算 + 单体调用):

```python
def pad_hpc_ops_block_scale(scale):
    k_pad = (k + _SCALE_K_ALIGN - 1) // _SCALE_K_ALIGN * _SCALE_K_ALIGN
    ...
x_q, x_scale = sglang_per_token_group_quant_fp8(x, HPC_OPS_BLOCK_SIZE)
output = hpc.fuse_moe_blockwise(x_q, x_scale, w13, w13_scale_inv,
                                w2, w2_scale_inv, topk_ids, topk_weights,
                                moe_ep_rank, global_num_experts)
```

## 性能声明

PR 正文数据(H200):

- Qwen3-30B-A3B-FP8(E=128, topk=8, hidden=2048, inter=768)MoE 层 kernel 时延,vs triton fused_experts:1 tok 356.4 → 89.9 us(**3.96x**)、8 tok 4.21x、32 tok 2.46x、128 tok 1.53x、512 tok 1.35x、1024 tok 1.16x、2048 tok 0.89x、4096 tok 1.08x。
- 同模型 e2e(bench_one_batch_server,input 1024/output 128,H200 TP1):bs=16 output 2126 tok/s vs auto(deep_gemm)2070 / triton 2083(+2.7%);bs=1/64 与 deep_gemm 基本持平。
- Hunyuan v3(tencent/Hy3-FP8,per-tensor FP8,192 experts,H200 TP4):decode output tok/s bs=16 950.5 → 1010.1(**+6.3%**)、bs=64 2544.3 → 2700.1(**+6.1%**);input tok/s bs=16 -12.5%、bs=64 -13.1%(大 M prefill 较慢,与 kernel 曲线一致)。
- GSM8K(1319 题):Qwen3 auto 0.940 / triton 0.937 / hpc_ops **0.944**;Hy3 default 0.942 / hpc_ops 0.942。kernel 三方对比:hpc_ops vs fp32-exact cosine 0.99974 / max_rel 0.024,与 triton 相同精度。

## 验证与回归风险

`test_hpc_ops_moe.py`(H200 上 2 passed、6 subtests,token ∈ {7,64,512},含 unsorted top-k ids,upstream HPC-Ops 测试只覆盖 sorted)+ `test_hpc_ops_runner_guard.py`(错误配置启动即 fail);CI PR Test 当时仍在跑(hourglass)。风险点:严格 opt-in、未装 HPC-Ops 时测试跳过,默认路径不变;2048 tok 段 kernel 反而慢于 triton(0.89x),大 prefill 场景不建议开启;per-tensor 路径要求 checkpoint 为静态激活 scale;依赖外部 `hpc` 包安装与 sm90+。

## 关联

前序/配套:#30540(HPC-Ops attention backend,第一期整合另一半)。同区域 MoE backend:#31470(flashinfer mega-moe)、#8036(flashinfer blockscale FP8)、#24816、#34967、#42557。上游:https://github.com/Tencent/hpc-ops 。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
