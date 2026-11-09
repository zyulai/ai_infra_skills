# RFC #6226: [RFC] Drop beam search support

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6226](https://github.com/vllm-project/vllm/issues/6226) |
| 作者 | WoosukKwon |
| 创建日期 | 2024-07-08 |
| 状态 | ⚪ closed (2024-07-22) |
| labels | RFC |
| 热度 | 👍 52 · 💬 42 |
| 分类 | Feature / Ecosystem |

## 一句话结论
提议移除 beam search 支持以降低系统复杂度,因社区强烈反对而撤回,vLLM 继续保留该功能。

## 提案动机
vLLM 支持 greedy、random、beam search 三种采样,其中 beam search 在每步动态创建和删除 top-k 分支,是最复杂的一种。beam search 传统上流行于翻译、摘要等 NLP 任务,但进入 LLM 时代已不常见,GPT、Gemini、Claude 等主要 LLM API 均不支持。尽管 beam search 正是 PagedAttention 的最初动机、vLLM 在此场景相对其他推理引擎有优势,其实现仍带来大量系统复杂度、阻碍潜在优化,却很少被使用,故提议移除。

## 提出的方案
- 第一步:从 sampler 与 output processor 中移除 beam search 及其直接相关代码;
- 第二步:在系统只需支持 greedy/random 采样的前提下进一步简化代码;
- 第三步:启用前述未来优化,包括——
  - 简化采样与输出处理代码;
  - block table 不再随 beam search 每步动态变化,消除 model runner 与 scheduler 间的同步,使其可异步运行;
  - 有望移除 SequenceGroup(parallel sampling 不需要 gang-scheduling,内存共享可交由 prefix caching 管理)。

## 优化点与收益
收益主要在复杂度与架构层面:采样/输出处理代码大幅简化、PagedAttention 的 block table 更可预测、SequenceGroup 有望移除,为 model runner 与 scheduler 异步化等后续优化扫清障碍。正文未给出量化数字。

## 现状与落地
已关闭,但提案并未执行:正文顶部原 TL;DR 被删除线划掉,代之以更新声明——"由于社区的强烈反对,我们决定重新考虑该提案,vLLM 将继续支持 beam search 直到另行通知",并感谢社区反馈。即本 RFC 因反对声而撤回,作者同时表示若社区有强需求可重新引入(原意移除时也如此承诺)。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
