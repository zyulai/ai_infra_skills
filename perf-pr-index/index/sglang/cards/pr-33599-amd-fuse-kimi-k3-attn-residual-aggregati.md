# PR #33599: [AMD] Fuse Kimi-K3 attn-residual aggregation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu |
| 合并 | 2026-08-05 |
| 分类 | Model-specific Fast Path |
| Label | jit-kernel |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见新增 attn_res_hip.py(212 行)并改 attn_residual.py |
| 链接 | https://github.com/sgl-project/sglang/pull/33599 |

## 一句话结论

为 Kimi-K3 在 ROCm 上把注意力残差聚合的 7 个 kernel launch 合成 1 个 Triton kernel,总吞吐提升 5-6%,ITL 提升 4-7%。

## 动机

Kimi-K3 的 attention-residual 聚合(score 对 bank 行打分、softmax、加权求和、out RMSNorm、pending residual 加法、bank snapshot 拷贝)原本在 ROCm 上走 2-kernel Triton pipeline(fallback 路径),加上 RMSNorm、residual add、bank 拷贝等共 7 次 launch。而 SM100 上已有单 kernel 快速路径,但依赖 TMA 与 tcgen05,AMD GPU 没有。decode 时 T 是 batch 大小,256-CU 的 GPU 上只有少量 CTA 在跑,瓶颈在单个 CU 的 load latency 与带宽,需要为这种场景重新设计 kernel 而非移植 SM100 版本。

## 优化点

新增一个 HIP Triton kernel(每 token 一个 CTA),把 `_score_kernel`、`_combine_kernel`、out RMSNorm、pending residual 加法(`_aggregate_fused_add` 原本单独 materialize)与 bank snapshot 拷贝合并为一次 launch:每层从 7 次 launch 降到 2 次。核心手法是把整个 bank 作为寄存器驻留 tile(`[next_pow2(nvb), next_pow2(H)]` fp32)一次读入,让打分与加权混合共享同一次读取,而不是像 SM100 online softmax 那样逐行做 block-wide 归约——后者的每个归约链环都是一次完整 HBM 往返(实测约 1.1us/row,NVB 线性)。代价是寄存器占用:`MAX_REGISTER_TILE`(8×8192)封顶形状,超出则寄存器溢出反而输给原 pipeline;在 MI355X、T=4 下实测 next_pow2(nvb)=8 时快 2.0x、16 时 1.2x、32 时仅 0.2x。K3 的 H=7168(一个 8192 的 masked tile)加 nvb≤8 恰好卡在预算内。kernel 还支持 `ow=None`(返回 pre-norm 混合值,供 `aggregate_stream()` 做 DSPARK aux capture)与 `addend=`(折入 residual 加法),这两点 TMA kernel 都做不到;取全局 max 再做指数使结果与 2-kernel pipeline bit-comparable。`_use_hip_fused()` 以 ROCm + 形状谓词门控,三个 dispatch 点接入;超预算形状与非 ROCm 平台路径完全不变。

## 关键改动

```text
python/sglang/kernels/ops/kimi_k3/attn_res_hip.py: 新增单 kernel ROCm 聚合实现与 register 预算门控
python/sglang/srt/layers/attn_residual.py: _use_hip_fused 门控 + _aggregate_hip 接入三个 dispatch 点
```

寄存器 tile 一次读入 bank:

```python
tile = tl.load(
    bank_ptr + t * stride_bm + offs_r[:, None] * stride_bb + offs[None, :],
    mask=mask_r[:, None] & mask[None, :],
    other=0.0,
).to(tl.float32)
# scoring 与 mix 复用同一 tile,不回 HBM
```

## 性能声明

PR 正文声明:total throughput 提升 5-6%,ITL 提升 4-7%(未注明具体模型/硬件/workload 之外的条件;kernel 注释中提到的测量点为 MI355X、T=4)。精度:gsm8k 1319 题 accuracy 0.962。

## 验证与回归风险

PR 未新增单元测试(diff 中无 test 文件),仅有 GSM8K 精度验证;CI 状态显示 PR Test (Base) 与 (Extra) 均失败标记(x),正文未解释。风险点:kernel 依赖寄存器预算谓词,形状超出 `MAX_REGISTER_TILE` 时回退原 pipeline,但 `_use_hip_fused` 对 `nvb` 的依赖意味着不同 batch/expert 配置会动态切换路径;`aggregate_stream` 中 HIP fused 路径返回 pre-norm 值时(ow=None)未传 out_norm,与原 `_mix_fused` 数值一致性依赖全局 max 的 softmax 实现;`fused_write` 条件扩展使 bank snapshot 写入也走 HIP 路径,依赖"无其他程序读 bank 行 NVB"的免同步假设。

## 关联

同区域优化:`attn_residual.py` 中已有的 SM100+ fast path(`_use_fast`,H=7168,TMA/tcgen05 版本)是本 PR 的 CUDA 对应物;DSPARK 相关聚合需求(`ow=None` 供 aux capture)与本批 PR #44200(水印 DSPARK 投机解码)可能在功能上相关,pack 未明确列出关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
