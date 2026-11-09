# PR #26588: Optimize Gemma4 H200 MoE and extend attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-06 |
| 分类 | Speculative Decoding (tier: major) |
| Label | run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files) |
| 链接 | https://github.com/sgl-project/sglang/pull/26588 |

## 一句话结论

Gemma4 H200 三项调优(MoE Triton 配置、Hopper extend-attention block size、小 batch QKV RMSNorm)使 TTFT -23.3%、TPOT -18.2%;两个 routing 融合优化因精度回退被撤销。

## 动机

Gemma4(`google/gemma-4-26B-A4B-it`,E=128/N=704 MoE)在 H200 上缺 tuned fused MoE Triton 配置,Hopper extend-attention 的 block size 对 `Lq=129..256` 区间不合理,MTP draft 路径的小 batch QKV RMSNorm 延迟也待压。优化刻意限定在 Gemma4/Hopper 路径,不影响其他模型族行为。

## 优化点

三项保留优化:(1) 新增 H200 Triton fused MoE 配置 JSON(normal 与 down projection 两份,按 M=1..1024 逐档 BLOCK_SIZE/GROUP_SIZE/num_warps/num_stages)。(2) Hopper extend-attention block size 细分:`Lq<=128` 用 (128,64),`Lq<=256` 用 (64,64),更大用 (32,64)——原来 `Lq<=256` 一律 (128,64),对 Gemma4 的 head 数造成 tile 浪费。(3) `_gemma_qkv_rmsnorm_kernel` 支持第二种 launch 形态 `BY_HEAD=True`:M<=256 时 grid 为 (M, total_heads),每 program 处理一个 token/head,提高小 batch(即 MTP draft)下的并行度,大 batch 保持原 per-token 循环形态。两项撤销优化:Gemma4 专用 topk+softmax+per-expert scale 融合 kernel,以及 Router 的 `rmsnorm(x, fused_scale, eps)` 单调用快路径——二者的浮点累积顺序与 eager 路径不 bit 等价(BF16 融合 RMSNorm、FP32 内部 softmax 归约树不同),导致 GSM8K MTP 分数 0.445→0.360,已分别 revert;作者注明恢复前提是做到 bit-equivalent(如 RMSNorm 接受 weight=ones、fused_scale 拆为独立 elementwise,softmax 逐位复刻 PyTorch 归约树)。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=704,device_name=NVIDIA_H200.json(+_down): H200 MoE tuned 配置
python/sglang/srt/layers/attention/triton_ops/extend_attention.py: Hopper Lq<=128/<=256/>256 分档 block size
python/sglang/srt/layers/gemma4_fused_ops.py: _gemma_qkv_rmsnorm_kernel 增加 BY_HEAD launch 形态(M<=256)
```

## 性能声明

单卡 NVIDIA H200,TP=1,`google/gemma-4-26B-A4B-it`,BF16,`--attention-backend triton`,random 1024/256,request rate 8,max concurrency 64,80 prompts;baseline `origin/main`(376635c1e)vs patched(27cb94c45,c2/c3 已 revert):output tok/s 1670.72 → 1725.50(+3.28%),p50 TTFT 96.56 → 74.04 ms(-23.3%),p50 TPOT 20.30 → 16.62 ms(-18.2%),median E2E 5295.91 → 4338.60 ms(-18.1%),total tok/s 8353.59 → 8627.50。

精度:GSM8K MTP CI(TP=2,`--enable-deterministic-inference`,NEXTN,200 例 5-shot,阈值 0.41):baseline 0.445 / patched 0.445(avg accept length 4.494)均通过;revert 前 PR head 0.360 不通过。二分定位:revert Router RMSNorm 融合 +0.035、revert MoE topk 融合 +0.020、两者同 revert 恢复 0.445;QKV RMSNorm 与 extend-attn/MoE config 两个提交对精度无影响。Standalone GSM8K(TP=1,无 MTP,8-shot,200 题):patched 0.41/0.495 vs baseline 0.465/0.445,run-to-run 波动约 5pp,无系统性回归。

## 验证与回归风险

依赖已注册的 Gemma4 MTP CI 测试(`test_gemma4_mtp_26b_a4b_extra::test_gsm8k_mtp`)做精度回归;block size 对 Lq=128/192/256/288 做了 sanity check;新配置 JSON 解析通过。风险:BY_HEAD 是新的 kernel launch 路径(仅 M<=256);撤销经验表明该模型 routing 路径对浮点顺序敏感,MTP `avg_accept_length`(4.48)会掩盖底层质量回退,单看接受长度不足以把关;CI Extra 失败(x)、Base 通过。

## 关联

同区域(卡片库中):#24696(Gemma4 fused Q/K/V RMSNorm 前作)、#30216(CPU fused QK Gemma norm);撤销的两个 kernel 若后续按 bit-equivalent 方案重做,将回到本 PR "Reverted optimizations" 一节的条件。
