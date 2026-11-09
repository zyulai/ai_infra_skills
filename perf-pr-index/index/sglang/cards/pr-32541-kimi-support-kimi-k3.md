# PR #32541: [Kimi] Support kimi-k3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-08-04 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | documentation, high priority, quant, amd, dependencies, Multi-modal, hicache, blackwell, npu, run-ci, jit-kernel, bypass-maintenance, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 未提供(pack 素材无增删行数字;完整 diff 超过 130 files,素材仅保留部分源码文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/32541 |

## 一句话结论

为 Kimi K3 提供 Day-0 支持,并附带视觉投影、CUDA IPC、MoonViT RoPE 等多模态路径性能优化。

## 动机

Kimi K3 是新发布的 KDA 线性注意力 + MoE 多模态模型,sglang 此前不支持。作者同时把多个 VLM follow-up(#31318、#31581、#31587、#31637、#31638、#31944)合入本分支,清理多模态推理路径上的冗余拷贝、多余同步与低效 kernel,使 K3 在 Day-0 即达到可用性能。

## 优化点

改动分两层。功能层:新增 KimiK3Config/KimiK3VisionConfig 模型与视觉配置,接入 KDA 线性注意力、Mamba radix cache 与 ReplaySSM 记账。调优层:(1) 消除 Kimi vision projection 的冗余 split/cat 拷贝;(2) 仅在选定 CUDA IPC transport 时默认开启 IPC pool-handle 缓存,复用有界 pool mapping,不额外占池;(3) packed MoonViT grid metadata 保留在 CPU 并复用 host 预计算的 attention metadata,消除逐请求 DtoH 拷贝;(4) vision RoPE 组件共享,连续布局的 K2.5/K2.7 Q/K 走 in-place JIT,K3 走既有 strided-QKV Triton 路径;(5) 为 M==1 decode 新增单 warp 的 moe_align_single_token JIT kernel,把 moe_align_block_size + count_and_sort_expert_tokens 两个 kernel(约 4.4us)合成一个约 1.5us 的 launch;(6) 新增 ROCm 可用的 top-k/top-p 概率 renorm Triton fallback。

## 关键改动

```text
python/sglang/srt/configs/kimi_k3.py: KimiK3Config / KimiK3VisionConfig,复用 KimiLinearConfig 作为 text_config
python/sglang/srt/arg_groups/overrides.py: KimiK3ForConditionalGeneration 的 attention backend / MoE runner / DCP 自动选择与默认值
python/sglang/srt/arg_groups/kimi_k3_hook.py: spec backend 默认值、CUDA graph 下强制关闭 symm-mem、KDA decode 用 triton
python/sglang/kernels/jit/csrc/moe/align_single_token.cuh (+ ops/moe/moe_align_single_token.py): M==1 decode 单 warp moe_align kernel
python/sglang/kernels/ops/sampling/renorm_triton.py: ROCm top-k/top-p renorm Triton fallback
```

单 warp 对齐 kernel 的机制(单 token 的 top-k expert id 互不相同,布局可直接构造):

```cuda
// Tiny moe_align_block_size for M == 1 decode: one warp replaces the
// moe_align_block_size + count_and_sort_expert_tokens kernel pair (~4.4us)
// with a single ~1.5us launch.
// For a single token the top-k expert ids are distinct, so the aligned
// layout is exactly: experts sorted ascending, one block per expert,
```

## 性能声明

- Projection packing 微基准,RTX 5090 BF16:1/2/4/8 image chunks 从 0.0242/0.0222/0.0231/0.0274 ms 降至 0.0140/0.0126/0.0126/0.0134 ms(**1.73x-2.05x**),仅隔离 post-projector packing 路径。
- CUDA IPC handle-cache A/B,4x H100 TP4,Qwen3.6-35B-A3B-FP8,每请求 4 张 seeded 随机图:output=1 burst 吞吐 7.55→20.43 req/s(**+170.6%**);output=1 rate=4 mean TTFT 496.30→97.70 ms(**-80.3%**);output=64 burst 吞吐 5.59→17.17 req/s(**+207.2%**);output=64 burst mean E2E 2007.29→560.14 ms(**-72.1%**)。
- MoonViT RoPE 微基准,NVIDIA B300 BF16(16 heads,head dim 72),按层 Q/K:256 tokens 47.58→7.60 us(6.26x);1,024 48.14→7.53 us(6.39x);4,096 63.55→7.84 us(8.10x);8,192 116.23→14.52 us(8.01x)。H200 TP8 早期 profile:launch 数 648→108,相关 GPU 时间 -5.802 ms,该 workload 下端到端在噪声内。
- Packed-grid metadata,H200 TP8:消除全部 12 个 Memcpy DtoH 事件,cudaStreamSynchronize 60→48;端到端 bench_serving 在噪声内,作者定性为同步清理而非独立吞吐声明。

## 验证与回归风险

验证:NVIDIA B300 上 260 passed、32 skipped、18 subtests passed,覆盖 Kimi K2.5/K3、multimodal transport、media error、dataset、server-args、vision metadata、vision RoPE 测试;fused RoPE 覆盖 1/480/5660/8360 random-token shapes、BF16/FP16、CUDA graph replay 与 K2.5 head-dim 72 parity;pre-commit 与 `git diff --check` 通过。风险:改动横跨多模态执行、CUDA IPC transport、symm-mem、Mamba/ReplaySSM 记账与 KDA 路径,可能波及 Kimi K2.5 等既有模型与 Blackwell/HIP 后端;CUDA graph 下自动关闭 symm-mem、DCP 强制 cutedsl_mla/trtllm_mla 后端等自动覆盖会改变用户的显式配置行为;新增 ROCm renorm fallback 只在 FlashInfer 不可用时生效(依 PR 语义,PR 未详述切换条件)。

## 关联

吸收 #31318、#31581、#31587、#31637、#31638、#31944;#31932 被本分支的 `any(...)` 断言取代;AMD Day-0 镜像细节见 #32548。同属 Model-specific Fast Path 分类。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
