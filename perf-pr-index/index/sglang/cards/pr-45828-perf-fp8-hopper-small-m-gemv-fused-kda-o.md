# PR #45828: [Perf] fp8: Hopper small-M GEMV, fused KDA output quant, CUDA grouped top-k, partial-block tails

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323(co-authored-by 文贤/得泽/文赋) |
| 合并 | 2026-11-04 |
| 分类 | MoE (tier: major) |
| Label | documentation, quant, run-ci, jit-kernel, run-ci-extra |
| Diff | pack 未提供 diffstat(diff 因超长截断,26 个文件未收录,4 个新增 .cuh 可见) |
| 链接 | https://github.com/sgl-project/sglang/pull/45828 |

## 一句话结论

为 Hopper block-FP8 decode 加 4 条 opt-in 快路径(GEMV/KDA 融合量化/CUDA grouped top-k/partial-block 尾部),bs=1 吞吐 +7.3%。

## 动机

从 Ling-3.1-flash day-0 支持工作拆出。Block-FP8 decode 在 Hopper 上有四个瓶颈:小 M 的 linear 跑在为大 batch 设计的 GEMM kernel 里;KDA 输出归一化与 FP8 量化是两个独立 kernel;512-expert sigmoid router 走通用 Triton top-k;K 维不是 128 量化块整数倍的权重完全无法使用原生 Triton/DeepGEMM 路径。本 PR 为每项加 opt-in 快路径,dispatch 有显式契约,默认行为不变。

## 优化点

四条路径。其一,Hopper small-M GEMV(`SGLANG_ENABLE_FP8_BLOCK_GEMV`,默认关):M≤8 的 CUDA block-FP8 GEMV,用 `mma.m16n8k32.e4m3` 指令 + K-split 两级寄存器流水,支持可选融合 side projection;另有 BF16 tiny GEMV(N≤64、M≤4、仅 SM90,`--bf16-gemm-backend gemv`)与 Bailing-class gate 形状的 CUDA router matvec(fp32/bf16 权重);dispatch 检查 device/dtype/layout/scale grid/shape,不满足走原路径。其二,融合 KDA 输出量化(`SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT`,默认关):KDA 输出的 gated RMSNorm 与 FP8 量化融合,直接产出 FP8 o_proj 消费的 `(q, scale)`;要求 target-verify batch 与 BF16,否则回退。其三,512 expert/8 组/group-top-4/topk≤8 sigmoid 路由的 CUDA grouped top-k,warp 级 ordered-float max + 4 组 heap 选择;dispatch 仅按形状(无模型检查),shared-slot 语义与 `moe_fused_gate` 契约一致(含 `renormalize=False` 与零分语义)。其四,partial-block FP8(按层 opt-in):K 跨量化块边界的 block-FP8 权重,原生 Triton MoE 的 partial-block quant/GEMM、`prequant_k_tail` dispatcher(DeepGEMM 仅在 TMA 对齐契约成立时:操作数连续、base pointer 与 K 被 16 整除)、两段 K 尾的融合 SwiGLU+quant。树内尚无模型启用(单独的 Ling3 model PR 跟进)。

## 关键改动

```text
python/sglang/kernels/jit/csrc/gemm/hopper_fp8_block_gemv.cuh: 新增,M<=8 block-FP8 GEMV(+可选 side projection)
python/sglang/kernels/jit/csrc/gemm/hopper_bf16_tiny_gemv.cuh: 新增,N<=64/M<=4 BF16 tiny GEMV(每输出行一个 block)
python/sglang/kernels/jit/csrc/gemm/router_gate_matvec.cuh: 新增,router gate matvec(固定求和顺序保证 TP rank 间 logits 一致)
python/sglang/kernels/jit/csrc/moe/bailing_moe_topk.cuh: 新增,512-expert sigmoid biased grouped top-k
其余 26 个文件(dispatch/融合量化/partial-block 路径等)未收录在 pack diff 中
```

```cpp
// Tiny-N bf16 GEMM for decode: out[M, N] = x[M, K] @ w[N, K]^T with M <= kMaxM and N <= 64.
// One block per output row; cuBLAS runs these shapes as split-K plus a separate reduce kernel.
constexpr uint32_t kTinyThreads = 128;
constexpr uint32_t kTinyVec = 8;  // bf16 per 16-byte load
template <uint32_t kMaxM>
__global__ void __launch_bounds__(kTinyThreads) hopper_bf16_tiny_gemv_kernel(
    bf16_t* __restrict__ out,
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w,
    const uint32_t M, const uint32_t N, const uint32_t K) {
```

## 性能声明

测试条件:H200,Ling-3.1-flash(FP8,TP8+EP8+NEXTN,fp32 router),3 轮 bs=1(2048/512)+ 2 轮 conc16(8192/1024),`--flush-cache`,seed 11。

| workload | main | 本 PR(env off) | 本 PR(flags on) |
|---|---|---|---|
| bs=1 output tok/s | 218.6 | 221.6(+1.4%) | 234.6(+7.3%) |
| bs=1 median TPOT (ms) | 4.351 | 4.191(−3.7%) | 3.981(−8.5%) |
| conc16 output tok/s | 1087.6 | 1087.8 | 1094.9(+0.7%,run noise 内) |

env off 的 +1.4% 来自形状驱动的 grouped top-k;flags-on 增益来自 GEMV 与融合 KDA 量化。trace(H200, bs=1 decode, flags on):510 个 dense FP8 GEMM 调用从 DeepGEMM `sm90_fp8_gemm_1d2d` 移到 GEMV kernel,189 对 KDA norm+quant 合并为融合 kernel;router matvec 在 fp32/M=4 为中性(1.41 vs 1.30 ms),PR 未声称其加速。GB300(SM103)上所有 SM90-gated 路径为 no-op,OFF/ON kernel 序列一致。低并发 decode 是这些 kernel 的收益区。

## 验证与回归风险

精度:H200 8x 全量 GSM8K(1319 题,temp 1.0, top_p 0.95, 每模式 2 轮)OFF 96.89%/stop 99.92% vs ON 97.04%/99.89%(±0.5pp 噪声内);GB300 4x TP4+EP4:OFF 96.82% vs ON 96.97%,stop 均 99.92%。Kernel 数值测试覆盖 GEMV/side-projection、router matvec、融合 KDA quant、grouped top-k(含 `renormalize=False` 与 zero-score)、block quant tail、Triton K-tail;partial-K linear 与 dequant-fp32 参考对齐(DeepGEMM K=160 与 Triton 回退 K=161)。Kimi-Linear-48B BF16 smoke(KDA 回退路径)100 题 GSM8K 93.0%。回归面:所有路径默认关闭、形状驱动 dispatch 需覆盖到其他同形状模型;GB300 上 SM90-only 套件干净跳过;CI Base/Extra/AMD 运行在 PR 时点均为失败状态(pack 记录)。

## 关联

前序/后续:Ling-3.1-flash day-0 支持工作(本 PR 自其拆出);K160 large-M 特化(H20-3e)与本 PR 无关、将单独跟进;Ling3 model PR 单独跟进。pack 未列出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
