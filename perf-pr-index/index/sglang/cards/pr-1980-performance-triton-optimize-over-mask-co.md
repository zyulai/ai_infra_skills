# PR #1980: [Performance, Triton] Optimize over mask compute to tl.load in fused_moe_kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HaiShaw |
| 合并 | 2024-11-11 |
| 分类 | Attention / Kernel (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/1980 |

## 一句话结论

fused_moe_kernel 偶数 K 块去掉逐次 mask 计算 + decode stage-1 补 ROCm launch 参数,Grok-1 decode 吞吐 +1.0%。

## 动机

Triton `fused_moe_kernel` 在 K 维循环的每次迭代里都对 A、B 的 `tl.load` 做 K 边界 mask(`offs_k < K - k * BLOCK_SIZE_K`),即使 K 恰好被 BLOCK_SIZE_K 整除也照算,这些 mask 计算是纯粹的循环内开销。同时 decode attention 的 stage-1 kernel(`_fwd_grouped_att_m_fwd`)尚未像 stage-2(PR #1845)那样设置 ROCm 专属 launch 参数,MI300x 上同样存在调优空间。两项改动合在一个 PR,针对 Grok-1(MoE + FP8)场景测量。

## 优化点

第一处:`fused_moe_kernel` 新增 `tl.constexpr` 参数 `even_Ks`,launch 前由 `invoke_fused_moe_kernel` 计算 `K = B.shape[2] - padding_size`(fp8 权重的 padding 剔除后)是否被 `config["BLOCK_SIZE_K"]` 整除。整除时(`even_Ks=True`)A 的 load mask 退化为只剩 `token_mask[:, None]`(token 维仍需屏蔽 padded token),B 的 load 完全去 mask;不整除时保留原双条件 mask。由于 `even_Ks` 是 constexpr,分支在编译期定死,循环内不再有每迭代的 K 边界比较。第二处:`_decode_grouped_att_m_fwd` 在 HIP 上给 `_fwd_grouped_kernel_stage1` 传入与 #1845 相同的 `extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}`,CUDA 路径不受影响。

## 关键改动

```text
python/sglang/srt/layers/fused_moe/fused_moe.py:
  fused_moe_kernel 增加 even_Ks constexpr;invoke_fused_moe_kernel 按
  K % BLOCK_SIZE_K 预计算并传入
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:
  _decode_grouped_att_m_fwd 的 stage-1 launch 增加 HIP extra_kargs
```

```python
if even_Ks:
    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
    b = tl.load(b_ptrs)          # K 整除时 B 完全免 mask
else:
    a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
```

## 性能声明

PR 正文:"~0.5% boost to prefill, ~1.0% boost to median decode throughput over Grok-1 with `b32/i1023/o256` settings"。即 Grok-1、batch 32 / input 1023 / output 256 设置下,prefill 约 +0.5%、decode 吞吐中位数约 +1.0%;未注明硬件(参考同序列 #1845 的测试环境为 MI300x,但本 PR 正文未写)与精度设置。

## 验证与回归风险

PR 正文无 accuracy 测试小节,diff 不含测试改动;验证依赖既有 fused_moe 与 decode attention 测试路径(pack 未提供)。回归风险:`even_Ks` 的判定源是 `B.shape[2] - padding_size`,若某权重的 padding 语义变化(K 实际有效维与该计算不一致),免 mask 分支会读越界或错位数据,属正确性风险;decode stage-1 的 ROCm 参数与 #1845 相同,风险同前(waves_per_eu/kpack 对非 MI300 卡未必最优);CUDA 路径不受影响。

## 关联

同作者前序 PR #1845(前一交易日合并)对 stage-2 kernel 做 ROCm launch 参数优化,本 PR 覆盖 stage-1 并叠加 fused_moe 的 mask 优化,两者构成同一 Grok-1/MI300x 调优序列;其余同区域编号 pack 未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
