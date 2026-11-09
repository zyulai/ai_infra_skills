# RFC #47684: [RFC]: FlashInfer NVFP4 KV serving on pre-SM100 GPUs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47684](https://github.com/vllm-project/vllm/issues/47684) |
| 作者 | lesj0610 |
| 创建日期 | 2026-07-06 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Quantization |

## 一句话结论
追踪让 FlashInfer 的 NVFP4 KV cache(--kv-cache-dtype nvfp4)在 pre-SM100(Ampere/Hopper)GPU 上可用的 PR 拆分、依赖关系与评审顺序。

## 提案动机
目标是让 FlashInfer 支撑的 NVFP4 KV cache 不依赖 Blackwell-only 路径,在 pre-SM100(主要是 Ampere/Hopper)系统上服务大上下文 Qwen3.6 与 Gemma 4 模型,从而把 FlashInfer attention/dequantization 支持与 NVFP4 约 3x 的 KV cache 容量收益结合起来。起点是 SM8x 硬件上的服务尝试,作者刻意把工作拆成通用 vLLM/FlashInfer 组件而非模型特定补丁。vLLM 已有 Triton 系量化 KV 路径(TurboQuant 4bit、int4 per-head),本 RFC 目标不同:让 FlashInfer CUDA 路径用上 NVFP4。

## 提出的方案
工作拆分到多个 PR(附状态与建议评审顺序):
- vLLM 侧:#46883 在 vLLM 消费 FlashInfer workspace sizing(Ready for review);#46963 pre-SM100 NVFP4 KV cache 更新核心路径(经 FlashInfer slot-mapping append,Ready for review);#47547 让 Gemma4 在 FA4 不可用时选择 FlashInfer(Ready for review);#46558 FlashInfer MM-prefix attention 支持(Draft);#47637 Qwen3-VL shape-specialized pos-embed JIT 修复(Draft)
- FlashInfer 侧:flashinfer-ai/flashinfer#3740 NVFP4 paged KV append with slot mapping;#3748 paged NVFP4 KV dequantization helper;#3822 SM8x-gated h512 NVFP4 FA2 prefill 优化
- 建议评审顺序:#46883 → #46963 → #47547 → #46558/#47637(后两个为 follow-up/依赖上下文,非核心路径阻塞)
- 部分 PR 保持 draft,是避免在核心件分诊前堆积更多 ready-for-review PR

## 优化点与收益
正文引用 NVFP4 的"roughly 3x KV cache capacity benefit"(未附具体测试条件);详细容量、MRCR 与 serving 基准表在 PR #46963 中。

## 现状与落地
🟢 open;反馈期至少一周,或到相关维护者对评审顺序与归属表态。作者本地验证目标为 RTX 3090 / SM86(Qwen3.6 与 Gemma 4 workload),无 Hopper 硬件,Hopper 侧未经其验证但意图仍是通用 pre-SM100 支持;明确不取代既有 Triton 量化 KV 选项,保持通用 vLLM 行为不变。

## 关联
- #46963、#46883、#47547(核心 vLLM PR)
- flashinfer-ai/flashinfer#3740、#3748

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
