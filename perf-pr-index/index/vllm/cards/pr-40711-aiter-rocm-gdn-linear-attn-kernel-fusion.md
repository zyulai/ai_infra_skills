# PR #40711: [Aiter][ROCm] gdn_linear_attn kernel fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tpopp |
| 合并 | 2026-05-08 |
| 分类 | CUDA Graph / Compiler(major,classify 结果,实为 GDN decode kernel 融合) |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 gdn_linear_attn.py、fla chunk kernel 与 _aiter_ops.py 改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/40711 |

## 一句话结论

GDN(Qwen3-Next 系)decode 路径接入 AITER 融合 Triton kernel,每层省 20–26 µs launch 开销,Qwen3-Next-80B MI355x 吞吐 +5.7%、TPOT −5.9%。

## 动机

GatedDeltaNet 的 decode 路径由多个独立 PyTorch op 与多个 Triton kernel 拼成:reshape、causal conv1d 单 token 更新、sigmoid gating、delta rule 递归、QKV 重排各自成 kernel。模型有 36 个 GDN 层 + 12 个 attention 层,每 decode 步的 kernel launch overhead 累计达每层 20–26 µs,是 TPOT 的可观组成。

## 优化点

接入 AITER 的优化 Triton kernel(来自 ROCm/aiter#2423):`fused_reshape_causal_conv1d_update_single_token` 把 reshape + conv1d + conv state 更新融合为一个 kernel,`fused_rearrange_sigmoid_gated_delta_rule` 把 rearrange + sigmoid gating + delta rule 状态更新融合。新增 `_forward_core_decode_fast` 只服务 decode-only 批次(无 spec、无 prefill),prefill 与 speculative-decode 逻辑从 decode-only 热路径剥离,缩小 CUDA-graph 区域;`prepare_gdn_attention_core_inputs` 与 `rearrange_mixed_qkv` 用 `torch.compile(fullgraph=True)` 把原来每个张量各一次 `contiguous()` 合并成单个 Triton 拷贝 kernel;`core_attn_out` 预分配输出缓冲穿过 chunked GDN kernel 管道,消除单独拷贝。`are_gdn_triton_kernels_available()` 集中探测,旧 AITER 构建优雅回退原路径;LoRA 投影路径不走 fast path。

## 关键改动

```text
vllm/model_executor/layers/mamba/gdn_linear_attn.py: 新增 forward_hip 与 _forward_core_decode_fast;torch.compile 合并 QKV 重排
vllm/_aiter_ops.py: 新增 are_gdn_triton_kernels_available() 集中探测
vllm/model_executor/layers/fla/ops/chunk.py / chunk_o.py: core_attn_out 预分配缓冲直写
```

关键片段(gdn_linear_attn.py,decode fast path 的两步):

```python
mixed_qkv_non_spec, b, a = gdn_aiter_fused_reshape_causal_conv1d_update_single_token(...)
gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
    ..., initial_state=ssm_state, inplace_final_state=True,
    cu_seqlens=..., use_qk_l2norm_in_kernel=True,
    core_attn_out=core_attn_out.reshape(-1))
```

## 性能声明

- 机制数字:融合移除每层 20–26 µs launch overhead,对应 TPOT 与吞吐 5–8% 改善(视 workload 与输入/输出比例)。
- E2E:Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,TP=1,AMD MI355x(gfx950),ROCM_AITER_FA,FULL_AND_PIECEWISE,random 1024/1024、并发 4、32 prompts:输出吞吐 456.52→482.55 tok/s(+5.7%),总吞吐 +5.7%,Mean TPOT 8.66→8.15 ms(−5.9%),P99 TPOT −7.8%,E2E latency −5.4%。基线为同镜像同配置无本 PR。
- 精度:gsm8k 5-shot,flexible-extract 0.8506→0.8605、strict-match 0.8097→0.8203,均在标准误内。

## 验证与回归风险

验证含 serve benchmark、gsm8k、ROCM_AITER_FA 后端启动检查、AITER 缺 GDN kernel 时的回退验证(`GDN_AITER_TRITON_AVAILABLE == False`)与 torch.profiler trace 的 kernel dispatch 检查;diff 未新增单元测试。风险面:fast path 依赖 ROCm/aiter#2423,旧 AITER 自动回退;仅 decode-only 批次生效,LoRA(`in_proj_qkv` 分离投影)不走 fast path;`gdn_attention_core` custom op 增加 `fast_kernel` 参数与 `a_or_z_out` 可变参数,改变 op 签名。

## 关联

同区域(GDN/线性注意力 decode):#31722(GDN decode 延迟优化)、#31167(移除 GDN attention decode 阻塞拷贝)、#43534(GDN 融合 kernel CPU 路径)、#61855(ROCm 融合 GDN MTP triton port)。依赖上游 ROCm/aiter#2423。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
