# RFC #6925: [RFC][Feature] Support Remote Prefill in PD Disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6925](https://github.com/sgl-project/sglang/issues/6925) |
| 作者 | rainj-me |
| 创建日期 | 2025-06-06 |
| 状态 | ⚪ closed (2025-08-14) |
| labels | (none) |
| 热度 | 👍 7 · 💬 2 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议在 PD 分离中支持 remote prefill 与条件式分离,让不适合 PD 的请求不必强制走分离路径。

## 提案动机
对 30B-70B 中型模型,并非所有请求都受益于 PD 分离,例如 input tokens ≤ 128、output tokens ≥ 512 的请求;这类场景下,条件式分离(conditional disaggregation)之上的 remote prefill 是一个「nice to have」的特性。

## 提出的方案
pack 正文较短:除 checklist 外仅一段动机与相关资源链接,未包含具体设计。正文只提出 remote prefill with conditional disaggregation 的方向,并引用 Dynamo 在 vLLM v0/v1 中实现 remote prefill 作为参考。

## 优化点与收益
正文未给出量化数字。

## 现状与落地
已关闭(2025-08-14)。正文未写明是否实现,落地情况需查关联 PR。

## 关联
- Dynamo 分离架构文档: https://github.com/ai-dynamo/dynamo/blob/main/docs/architecture/disagg_serving.md
- vLLM PR #17751: https://github.com/vllm-project/vllm/pull/17751
- vLLM PR #16677: https://github.com/vllm-project/vllm/pull/16677

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
