# RFC #6378: [RFC]: A Graph Optimization System in vLLM using torch.compile

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6378](https://github.com/vllm-project/vllm/issues/6378) |
| 作者 | bnellnm |
| 创建日期 | 2024-07-12 |
| 状态 | ⚪ closed(关闭于 2025-04-13) |
| labels | RFC, stale |
| 热度 | 👍 8 · 💬 10 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议基于 torch.compile 自定义编译器构建图优化系统,把侵入模型定义的性能优化与模型代码解耦。

## 提案动机
Neural Magic 为 Torch Dynamo 编写自定义编译器,在 vLLM 内建立一个可编写图变换的系统,核心目标是把高层模型定义与性能攸关的低层决策解耦——尤其适合那些对模型定义侵入性强、破坏抽象、跨层边界、或并非普遍有效/有用的优化;若把它们写进模型定义,新增模型会变得困难。初始优化三项:把 quantize 操作融合进 LayerNorm 内核(fp8/int8、静态与动态量化);融合包含 GEMM、SiLU、Mul、quantize 的 MLP 段;把 Gemm+AllReduce+LayerNorm+Gemm 重写为 Fused Gemm-ReduceScatter + LayerNorm + Fused AllGather Gemm,以利用 ByteDance 的 Flux 内核。作者的理念:图优化系统是 vLLM 开发者的能力乘数;torch.compile 至少在线性层上不太可能取代自定义内核;不应把 torch.compile 当黑盒;系统应由 vLLM 掌控并与 Torch Inductor 良好互操作;保持轻量——vLLM 不应变成图编译器。

## 提出的方案
正文 Proposed Change 一节仅列出关联 issue:#6377、#9886;系统设计细节见正文链接的 Google Docs 详细提案。该系统虽作为 Torch Dynamo 内的自定义编译器运作,但更适合理解为 vLLM 的优化系统而非编译器:不做逐层 IR 的纵向编译栈,而是务实改进 vLLM 的自定义内核生态。

## 优化点与收益
正文未给出量化数字。

## 现状与落地
已关闭(2025-04-13,label 含 stale)。正文未记录最终落地情况,需查关联 PR #6377、#9886。

## 关联
- 详细提案(Google Docs):https://docs.google.com/document/d/1CvbJ0LOotlfTjR6RmlQKLO4zcEvN2deoDacSjQ31Xiw
- 关联 issue #6377(正文 Proposed Change 所列)
- 关联 issue #9886(正文 Proposed Change 所列)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
