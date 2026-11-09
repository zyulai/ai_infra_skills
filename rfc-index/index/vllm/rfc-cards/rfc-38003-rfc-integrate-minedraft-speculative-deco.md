# RFC #38003: [RFC]: Integrate MineDraft speculative decoding into vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38003](https://github.com/vllm-project/vllm/issues/38003) |
| 作者 | Sebastian-dong |
| 创建日期 | 2026-03-24 |
| 状态 | ⚪ closed (2026-06-05) |
| labels | RFC |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
提议把 MineDraft 的 batch 并行投机解码接入 vLLM:两个 batch 交替做 drafting 与 verification,把 drafting 延迟藏进重叠里。

## 提案动机
vLLM 现有投机解码把 drafting 与 verification 顺序执行,drafting 位于关键路径上,限制了可达加速。MineDraft 引入 batch 并行范式:在不同 batch 之间重叠 drafting 与 verification, effectively 隐藏 drafting 延迟,且无需重训模型即可提升吞吐与延迟。

## 提出的方案
- 两 batch 调度机制:Batch A 做 verification、Batch B 做 drafting,每个解码步交替两者角色。
- 关键集成点:Scheduler 负责 batch 级协调;LLMEngine 提供重叠执行流;KV cache 避免为 draft-only 请求超额分配。
- 兼容 continuous batching、PagedAttention 与现有投机解码方法;作为可选解码模式实现,不影响现有工作流;设计被接受后拆成多个小 PR。
- 完整设计见正文附的 MineDraft.md 设计文档。

## 优化点与收益
正文引用论文初步结果:Throughput up to +75%,Latency up to -39%(来自 MineDraft 论文,正文未附在 vLLM 内的复现数字)。

## 现状与落地
已关闭(2026-06-05);正文与结尾看不出是否实现,落地情况需查关联 PR。

## 关联
设计文档 MineDraft.md https://github.com/user-attachments/files/26212782/MineDraft.md

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
