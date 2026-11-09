# RFC #40902: [Roadmap] DeepSeek V4

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40902](https://github.com/vllm-project/vllm/issues/40902) |
| 作者 | ivanium |
| 创建日期 | 2026-04-26 |
| 状态 | ⚪ closed (2026-06-23) |
| labels | RFC, deepseek, keep-open |
| 热度 | 👍 32 · 💬 12 |
| 分类 | CI / Infra / Process |

## 一句话结论
DeepSeek V4 支持与优化的官方路线图:核心实现已合入 main,围绕低延迟、PD、超长上下文等方向持续迭代。

## 提案动机
该 issue 是 DeepSeek V4 在 vLLM 中的实施路线图:集中跟踪实现进度(实现统一基于 PR #40860,并要求后续优化工作叠加在 main 之上)、指引贡献者认领各方向任务,性能表现则以外部基准 InferenceX 仪表盘为准。

## 提出的方案
路线图按八个方向组织:
- 核心模型支持:FP4 Indexer 与 MegaMoE 初始支持均已在 PR #40860 完成(MegaMoE 后续在 #40833 继续);NVFP4 支持待做;
- 低延迟优化:已完成 C4A/C128A 的 multi-stream 4 GEMM(#41061)、更快的 fp8 group 量化 kernel(#41326)、norm 与 router 融合(#41263);待做:fast topk kernel、Indexer topk + page table 变换融合(#41105)、面向低 BS 的专用 pre-Attention GEMM;
- PD + 高吞吐:Nvlink_one_sided a2a 已支持 bf16 与 mxfp8 dispatch(#40960);FP8 quantized combine 待 flashinfer 更新 kernel(无已知 ETA);PD 调参与配置已完成;
- 超长上下文:chunked pipeline parallelism、prefill context parallelism;
- 运行时与并行:Model Runner V2 集成、MTP 优化、pipeline parallelism 支持;
- Kernel 集成:主注意力与 indexer 注意力的 paged prefill kernel(FlashInfer?FA4?)、调研 DeepEP V2、TileKernels(https://github.com/deepseek-ai/TileKernels)相关集成;
- KV cache:PD + CPU offloading(#39654)、分布式 KV offloading;
- 硬件支持:Hopper 已完成(#40860),SM120、AMD 待做。

## 优化点与收益
优化覆盖低延迟(multi-stream GEMM、kernel 融合、专用低 BS GEMM)、高吞吐与 PD 分离(a2a dispatch、PD 调优)、超长上下文(并行化)与 KV cache offloading 等方面;量化数据以 InferenceX(https://inferencex.semianalysis.com/inference)仪表盘为准,正文未给出量化数字。

## 现状与落地
已关闭(2026-06-23):正文明确核心实现 PR #40860 已合入 main,路线图上 FP4 Indexer、MegaMoE 初始支持、Hopper 支持及多项低延迟/PD 优化均已勾选完成;NVFP4、超长上下文、部分并行与 kernel 集成项在关闭时仍未完成,后续落地需查各关联 PR。

## 关联
- https://github.com/vllm-project/vllm/pull/40860 (核心实现,已合入 main)
- https://inferencex.semianalysis.com/inference (性能仪表盘 InferenceX)
- https://github.com/vllm-project/vllm/pull/40833 (MegaMoE 后续)
- https://github.com/deepseek-ai/TileKernels (TileKernels 仓库)
- https://github.com/vllm-project/vllm/pull/39654 (PD + CPU offloading)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
