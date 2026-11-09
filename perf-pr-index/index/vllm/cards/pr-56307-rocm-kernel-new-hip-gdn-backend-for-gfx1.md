# PR #56307: [ROCm][Kernel] New "hip" GDN backend for gfx1151

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @roberteg16 |
| 合并 | 2026-11-03 |
| 分类 | Attention / Kernel(major) |
| Label | performance, rocm, ready, ci/build |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含新 HIP kernel、benchmark,另有 11 个文件因篇幅被省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/56307 |

## 一句话结论

为 RDNA3.5 (gfx1151) 新增单 HIP kernel 的 chunked GDN prefill 后端,替代经 DDR 传递中间结果的 Triton kernel 链,公共 API 计时中位提速 2.19–3.05x,E2E TTFT −7–8%。

## 动机

GATED_DELTA_NET prefill 在 RDNA3.5 上是一串 Triton kernel,通过 DDR 交换中间结果:2048 token 时每层要写回并重读 32 MiB。把 chunked delta rule(WY 表示/UT transform)做成单个 kernel:一个 block 走一个 (head, sequence) 的所有 chunk,state 常驻寄存器、每 chunk 中间量放 LDS。

## 优化点

新增 csrc/rocm/rdna35_gdn_chunked.cu:chunk=32 token,恒等式 v_new = Tinv (V beta − (K beta exp(g_cs)) S) 使 h/w/u 永不物化,一个 [32][128] 缓冲原地复用,整块装入 64KB LDS;输入到输出全程 fp32(原 Triton 路径把 h/w/u/Tinv/v_new round 到 bf16),对 fp64 逐 token 参考的相对 RMS 误差为原路径的 0.53x,终态以 fp32 输出。微架构层面:DPP cross-lane 归约避开 LDS、decay 转 base-2 后用 v_exp_f32、gfx1151 专属寄存器水位(16 waves/SIMD)与 CU mode。以新 GDN prefill backend "hip" 暴露(triton/flashinfer/cutedsl/aiter_flydsl 之外):gfx1151 上 auto 自动选中;gfx1150–gfx1153 可用 --gdn-prefill-backend hip 显式启用、op 未编译时 fail closed;仅支持 bf16/fp16、head dim 128,内核无法服务的调用逐次回退 Triton;Triton/FLA 路径不动。

## 关键改动

```text
csrc/rocm/rdna35_gdn_chunked.cu(新增): 单 kernel chunked GDN prefill(RDNA3.5 专属)
csrc/rocm/ops.h: rdna35_gdn_chunked op 声明
vllm/model_executor/layers/mamba/ops/hip_gdn_chunk.py(chunk_gdn_hip_fwd wrapper,
  diff 被截断,由 benchmark 代码引用证实)
benchmarks/kernels/benchmark_gdn_chunk.py(新增): correctness-first 双路 benchmark
```

## 性能声明

AMD Radeon 8060S(gfx1151),公共 API(含 wrapper 分配)triton.testing.do_bench、cold L2:按 head shape 的 median speedup——16/16(Qwen3.5-0.8B/2B)2.19x;32/16(Qwen3.5-4B/9B/35B、Qwen3-Next)2.81x;48/16(Qwen3.5/3.6-27B)2.48x;64/16(Qwen3.5-122B/397B)3.05x;单 case 最大 9.938x(32×2048 tokens)。E2E Qwen3.5-9B prefill(16 req/点):TTFT 2048 in 1231→1132 ms(−8.0%)、8192 in 5754→5340 ms(−7.2%);prefill 吞吐 1.087x / 1.077x。GSM8K(Qwen3.5-9B,400 题 5-shot)0.873 vs Triton 0.870。

## 验证与回归风险

tests/kernels/mamba/test_gdn_chunked_rocm.py 39 passed(独立 fp64 逐 token 参考,含 ratio-3 heads、无初始态、跨对齐/非对齐续跑、adversarial gates、fp16);test_gdn_rocm_layout_dispatch.py 27 passed(backend 选择、生产 ChunkGatedDeltaRule 解析到 hip);每个 benchmark case 先过数值检查再计时。仅 gfx1151 实测,gfx1150/1152/1153 仅编译测试;其他 head dim/dtype 解析到 Triton,风险面为新增 backend 的分发逻辑。

## 关联

GDN prefill backend 家族(triton/flashinfer/cutedsl/aiter_flydsl)新增 "hip" 成员;pack 未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
