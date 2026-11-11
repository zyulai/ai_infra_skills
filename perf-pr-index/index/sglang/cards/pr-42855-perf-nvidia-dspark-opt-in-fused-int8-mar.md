# PR #42855: [Perf] [NVIDIA] dspark: Opt-in fused int8 Markov walk kernel for bs <= 64 on Hopper

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @tie-pilot-qxw |
| 合并 | 2026-11-10 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(超长 diff 被截断:保留新文件 dspark_markov_walk.cuh 前半,另 12 个文件略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/42855 |

## 一句话结论

DSpark 的串行 Markov walk 换成 opt-in 的 int8 persistent cooperative kernel,H100 上 walk 本身快 3.3-6.5x,端到端输出吞吐 +1.2-9.0%。

## 动机

DSpark 一次并行 forward 提出 γ 个 draft tokens,再用串行 Markov walk 逐步修正:第 k 步 `logits_k = base_k + W2 · W1[x_{k-1}]`(rank 256,全词表)后 argmax 或采样。folded draft graph 把这个 walk 跑成 γ 个独立 step,每步一次 bf16 GEMM 从 HBM 重读整个 W2(Qwen3 的 151,936 词表时 78 MB)再加一次 `MarkovSampleStep`。H100 上 γ=7 的 walk 在 bs=1 花 269 µs、bs=64 花 453 µs;γ=16、bs=1 花 608 µs,是 decode 的显著开销。

## 优化点

一个 JIT kernel(仅 sm_90a)覆盖 bs 1-64 的全部 draft graph bucket,机制四件套:(1) 一次 persistent cooperative launch 跑完所有 γ 步(网格必须 co-resident,驱动否则拒绝启动);(2) W2 的 int8 副本摊到各 SM——每 CTA 11 个 64 行 tile 驻留 shared memory,其余经 2-slot TMA ring 从 L2 流式读取,18 tiles/CTA 覆盖 152,064 行(Qwen3 的 151,936),248,320 需 30;(3) bias 用精确 int32 tensor-core 点积(`wgmma` m64n64k32 s8,W1 行为 register A operand、W2 为 smem B operand,32 requests/block、两个 block 落在 4 warpgroups),logit 经 row scale 后一次性舍入 bf16;(4) 每步 token 由跨 CTA 的 atomic `{key, count}` 交换选出(每 request 一次,平局取最小行号,与 torch.argmax 一致)。greedy 直接对 bf16 logits 取 argmax;T>0 时从 kernel 写出的同一 bf16 logits 采样(8-row item 按 Gumbel-max,item 内 inverse CDF,噪声用 MUFU log/exp 近似)。集成在 dspark_draft_sampler.py,由 `SGLANG_DSPARK_OPT_INT8_MARKOV_WALK=1` 开启;TP>1 时每 rank 跑整个 walk,rank 0 的 draft tokens 每轮广播一次(默认 walk 每步广播);walker 开启时 online weight updates 被拒绝;GPU 非 sm_90、W2 词表 TP 分片、markov head bias 非精确 GEMV(gated/RNN/Nemotron/DeepSeek-V4 head)、rank 非 256、词表超 32 tiles/CTA、γ>256 等情形启动即失败并给出原因。

## 关键改动

```text
python/sglang/kernels/jit/csrc/speculative/dspark_markov_walk.cuh: 新文件,int8 walk kernel(wgmma s8、W2 tile 驻留+L2 流式 ring、跨 CTA key/count 交换、Philox 噪声)
python/sglang/kernels/ops/speculative/dspark/markov_walk.py: MarkovWalker(W1/W2 分块量化与布局、JIT 加载、graph capture 前预热)
python/sglang/srt/speculative/dspark_draft_sampler.py: SGLANG_DSPARK_OPT_INT8_MARKOV_WALK=1 时接入 folded draft graph
jit/include/sgl_kernel/utils.cuh: LaunchKernel::enable_cooperative()
```

```cpp
//   for every request b:  prev_b = anchor_b;  for k < K:  logit = bf16(base[b, k] + W2 W1[prev_b]);
//                         tok[b, k] = argmax(logit)  or  a sample of softmax(logit / T_b);  prev_b = tok[b, k]
```

## 性能声明

H100 80GB SXM(132 SMs),CUDA 13.1,`bench_dspark_markov_walk.py`,CUDA graph replay、cold L2,µs/call:walk alone 比 default walk 快 3.3-6.5x。V=151,936、γ=7、bs=1:62.8 vs 268.6(greedy);bs=64:107.0 vs 453.0;γ=16、bs=1:124.9 vs 607.6;V=248,320、γ=15、bs=16:229.3 vs 1130.8。端到端:六个 target/drafter 对(DeepSpec eval mix,`--speculative-algorithm DSPARK`,FA3),客户端输出 tok/s 提升 1.2-9.0%——最高为 Qwen3-4B + `shreybirmiwal/Qwen3-4B-DSpark-b16` 的 T=1/c=1 +9.0%,最低 Qwen3-14B 的 +1.2%;各行的 accept length 95% CI 覆盖 0,即不变。驻留内存 114 MiB(Qwen3 词表)/ 188 MiB(248,320);首次 JIT build 约 12 s,此后 server 启动 <1 s。

## 验证与回归风险

test_dspark_markov_walk.py(17 passed,合成权重、18/30 tiles per CTA、单/双 32-request block):greedy 对 kernel int8 算术逐 bit、对 float64 dequant 参考仅 near-ties 不一致;T>0 行 corrected logits bit-exact;采样 χ² 检验(8,000 draws/案,p>1e-4);NaN/inf 行;每次 CUDA graph replay 全新噪声;γ=40 连续三轮;host 侧参数检查。E2E test_dspark_int8_markov_walk.py(Qwen3-4B + block7 drafter,extra tier):greedy 输出与 default walk 完全一致;并发 16 的 sampling acceptance length 在 default 的 10% 内(4.691 vs 4.675;TP2 4.662/4.571)。精度边界:greedy 不变;采样每 draft token total variation 约 1e-5;W2 int8 相对误差 0.58%,near-ties 可翻转,drafts 与默认 walk 非 bit-identical。风险:仅 sm_90a 且 bs≤64 bucket;248,320 词表需至少 122 SM;walker 开启时 online weight update 被拒。

## 关联

无(素材正文未引用其他 PR 编号;与本卡同区域的 DSpark walk/sampling 改动可在索引中按 Speculative Decoding 分类检索)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
