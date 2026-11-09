# RFC #45668: [Roadmap] Minimax M3

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45668](https://github.com/vllm-project/vllm/issues/45668) |
| 作者 | gau-nernst |
| 创建日期 | 2026-06-15 |
| 状态 | ⚪ closed (2026-08-01) |
| labels | RFC |
| 热度 | 👍 19 · 💬 10 |
| 分类 | CI / Infra / Process |

## 一句话结论
Minimax M3 模型支持路线图:跟踪 FP8/NVFP4 量化路径、MXFP8 MoE 集成与内核优化等落地项。

## 提案动机
正文是一份路线图清单,动机部分仅指向初始模型支持 PR #45381,未展开问题背景陈述(此处如实转述,不臆测)。

## 提出的方案
- **Key features**:FP8 indexer(@zyongye,#45892,已完成);FP8 sparse GQA(@gau-nernst,#45744,已完成);FlashInfer TRTLLM MXFP8 MoE(vllm PR #45723,依赖 flashinfer PR #3504,标注 Pending);NVFP4 支持(含 NVFP4/MXFP4 Indexer,已完成);Heterogeneous PD(@ivanium,未勾选);FP8 KV cache(已完成)。
- **Kernel optimizations / integrations**:BF16 indexer(来自 MSA,未勾选);尽量为尽可能多的内核启用 PDL(@jeejeelee,未勾选)。

## 优化点与收益
正文未给出量化数字。定性收益:通过 FP8/NVFP4 量化路径(indexer、sparse GQA、KV cache)与内核级优化(PDL 等)提升 Minimax M3 在 vLLM 上的性能与部署能力。

## 现状与落地
已关闭(2026-08-01)。清单中多数关键特性已勾选完成(FP8 indexer、FP8 sparse GQA、NVFP4 支持、FP8 KV cache);FlashInfer TRTLLM MXFP8 MoE 依赖上游 flashinfer#3504 且当时 Pending;Heterogeneous PD 与 BF16 indexer、PDL 两项内核优化未勾选。注:pack 正文较短(仅路线图清单),落地细节需查关联 PR。

## 关联
- [PR #45381](https://github.com/vllm-project/vllm/issues/45381)(初始模型支持)
- [PR #45892](https://github.com/vllm-project/vllm/pull/45892)(FP8 indexer)
- [PR #45744](https://github.com/vllm-project/vllm/issues/45744)(FP8 sparse GQA)
- [PR #45723](https://github.com/vllm-project/vllm/pull/45723)(vLLM MXFP8 MoE)
- [flashinfer#3504](https://github.com/flashinfer-ai/flashinfer/pull/3504)(上游依赖)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
