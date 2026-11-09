# PR #31888: [DSA] Q8KV8 FP8 Sparse Prefill on GLM-5.2 & DeepSeek-V3.2: Q8-Path & Shared-Path Optimizations

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang |
| 合并 | 2026-07-30 |
| 分类 | Quantization(tier: major) |
| Label | documentation, quant, deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计);diff 因超长裁掉 27 个文件,仅保留部分 |
| 链接 | https://github.com/sgl-project/sglang/pull/31888 |

## 一句话结论
对 DSA 稀疏 prefill 做 Q8KV8 fp8 路径与共享管线两层优化,GLM-5.2 净收益 saturated 吞吐 +13.1–17.5%、c=1 TTFT +14.4–20.0%。

## 动机
DeepSeek-V3.2 的 Q8KV8 fp8 sparse-MLA prefill 路径(#25751 kernel、#30514 DSA 集成)仅按 backend/KV-dtype/SM90 分派,模型无关;本 PR 在 GLM-5.2(`GlmMoeDsaForCausalLM`)上做端到端验证(64 q-heads 恰为 kernel 原生 tile,零新增集成代码),并贡献 7 项优化、2 个修复与推荐 flag 配置。此前的瓶颈包括:KV-gather 每 (token,128-elem slice) 一个 program 且要 `torch.zeros` 预清零(112us/call)、q-prep 走 cublas bf16 bmm + 独立 concat-cast 两趟、以及上游 JIT-DeepGEMM 在 bs=1 的 DLPack stride 崩溃。

## 优化点
两个独立阶段,分别报告。Stage 1(共享、backend 无关):NVLS + GATHERV(上游默认关闭,合计 sat +6.9–8.0%)+ DLPack stride shim(修复 JIT-DeepGEMM bs=1 崩溃,即 "JIT-DeepGEMM restore")+ fp8 DP-gather wire。Stage 2(q8-only,全部 bit-exact 门控):向量化 16B KV-gather + fused pad-zero + 持久缓冲(7.2x,112→15.6us/call);born-fp8 q 融合 absorbed-bmm+concat+cast(GLM 1.31x / DS 1.68x);per-row topk_length early-exit + backscan;persistent kernel-output buffers;fused non-prefix KV prep(3 kernel+1 alloc → 1);SM90 WGMMA q-prep kernel(2 CTA/SM + double-buffered B,较 Triton 最优 1.25x GLM / 1.16x DS)。所有行为可选项保持 opt-in(默认 off),bit-exact 内部改写与 bugfix 常开;另记录多项负结果(grouped-GEMM q-prep、split-topk 等)。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py: 向量化 16B KV-gather + fused pad-zero + cached buffer
python/sglang/kernels/jit/csrc/qprep_bf16_fp8_sm90/: 新 SM90 WGMMA born-fp8 q-prep kernel(K=128/192)
python/sglang/kernels/ops/attention/qprep_bf16_fp8_sm90.py: Python wrapper(dtype/shape/alignment 校验)
benchmark/kernels/deepseek/benchmark_q8kv8_kv_gather.py / benchmark_q8kv8_q_prep.py: bit-exact 校验型 microbenchmark
```

## 性能声明
8xH200、TP8/EP8/DP8、两臂均 fp8 KV cache、`sglang.bench_serving`(random dataset、range-ratio 1.0、flush cache)。Stage 1(bf16 FlashMLA 基线本身):GLM-5.2 sat +8.8–10.3%、c=1 TTFT +4.4–6.3%;DeepSeek-V3.2 sat +4.3–6.7%(TTFT ~flat +0.3–0.9%)。Stage 2(q8 vs 已优化 q16,q8 全胜 14/14 行):GLM sat +3.4–6.8%、TTFT +9.6–13.6%(3.5k TTFT 361ms);DS sat +5.0–10.1%、TTFT +8.1–20.9%。净效果(oo bf16 → 优化 q8):GLM sat +13.1–17.5%、TTFT +14.4–20.0%;DS sat +9.5–17.4%、TTFT +8.4–21.3%(3.5k–160k context)。局限:单节点 8xH200、单一并行拓扑、合成定长 workload,多节点与 PD 分离未测。

## 验证与回归风险
kernel pytest 21→42(42/42 通过,含 bitwise topk-length/backscan sweep 与 WGMMA q-prep suite);每个 lever 附 gsm8k 探针门(0.94–0.97);c=512 压力套件 3/3;DeepSeek-V3.2 全栈 gsm8k 0.951 vs 0.957(≈1σ);forced-sparse MMLU(N=14,042)q8 0.8146 vs q16 0.8120;LBV2 最终栈 q8-vs-q16 Δ −0.8~−1.0%(≈1σ)。风险:唯一有损 lever(fp8 DP-gather wire)默认关闭,其 LBV2 代价约 1.8%(GLM);q8 kernel dispatch 仅 SM90;多数优化为 opt-in env,默认路径收益有限。

## 关联
路线图 #25746;前序 #25751(kernel)、#30514(DSA 集成);上游修复依赖 #30512(fused top-k v2 tie-overflow IMA)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
