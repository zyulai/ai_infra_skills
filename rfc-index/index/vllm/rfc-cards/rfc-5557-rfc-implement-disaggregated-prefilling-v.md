# RFC #5557: [RFC]: Implement disaggregated prefilling via KV cache transfer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5557](https://github.com/vllm-project/vllm/issues/5557) |
| 作者 | KuntaiDu |
| 创建日期 | 2024-06-14 |
| 状态 | ⚪ closed (2025-06-21) |
| labels | RFC, stale |
| 热度 | 👍 38 · 💬 19 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议引入 communicator 与 KV database 两个抽象,统一支持 vLLM 实例间 KV cache 传输与存储。

## 提案动机
越来越多用例需要在 vLLM 实例之间传输 KV cache,或把 KV cache 存储起来供日后使用。典型场景一是分离式 prefilling:需要把 KV cache 从 prefilling 实例传输到 decoding 实例;二是用户查询固定的一批长文档(如软件手册、内部文档)时,GPU 内存加 CPU 内存可能装不下全部文档的 KV cache,希望将其存储到外部、按需搬上 GPU。

## 提出的方案
- 引入两个新抽象:communicator 与 KV database,整体工作流为 `vllm <--> communicator <--> KV database`;
- communicator 负责把数据从 src 传输到 dst,src 与 dst 既可以是 vLLM 中的 KV block,也可以是数据库中的条目;
- KV database 是以 hash(由自动前缀缓存生成)为 key、对应 KV cache tensor 为 value 的数据库;
- 作者同时列出该框架内一系列有挑战性的开放问题:如何利用 NVLink 等基础设施加速 KV cache 传输、如何对传输做合理流水线化、如何保证 communicator 工作期间相关块不被换出、是否在传输时压缩 KV cache(若压缩,选什么算法、由谁执行)。

## 优化点与收益
目标是为分离式 prefilling 与 KV cache 外存/按需加载提供统一的基础设施抽象,提升实例间协作与显存利用效率,并为利用 NVLink 等高速互连加速传输留出优化空间。正文未给出量化数字。

## 现状与落地
已关闭,落地情况需查关联 PR。正文显示作者当时处于设计征集阶段:设定数周反馈期,并 CC 了多位核心维护者征求对该抽象设计的意见(询问该抽象能否在各用例下达到最优性能)。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
