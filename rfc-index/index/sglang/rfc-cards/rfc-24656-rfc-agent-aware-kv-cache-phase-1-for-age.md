# RFC #24656: [RFC] Agent-Aware KV Cache Phase 1 for Agentic Workloads

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24656](https://github.com/sgl-project/sglang/issues/24656) |
| 作者 | stmatengss |
| 创建日期 | 2026-05-08 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 13 · 💬 11 |
| 分类 | KV Cache / Memory |

## 一句话结论
为 agentic 工作负载加 agent_hints 元数据通路与 agent_aware 淘汰策略,Phase 1 只打地基。

## 提案动机
LRU、LFU、SLRU 等现有 KV cache 淘汰策略基本把请求当独立个体处理,对标准 chat/completion 没问题,但错过了 agentic 工作负载的复用机会:planner/tool/worker 步骤间重复的 prompt 前缀、父子步骤关系、工具调用后的短期复用、workflow 级 TTL/复用意图。缺乏 workflow 感知信号时,很可能被后续步骤复用的缓存条目会被过早驱逐。

## 提出的方案
Phase 1 引入(刻意是基础/原型阶段,不是完整的 workflow 感知缓存系统):
- OpenAI 兼容请求新增可选 agent_hints 字段,并在内部请求管线中传播。
- 调度器做轻量 workflow DAG 跟踪;radix cache 节点标注 workflow 元数据。
- 新增可选 radix 淘汰策略 agent_aware。
- agent_hints 初始字段:workflow_id、agent_id、step_id、step_index、total_steps、parent_step_id、children_step_ids、tool_name、expected_tool_duration_ms、cache_ttl_ms、shared_prefix_hash、reuse_hint。
- 向后兼容:字段可选、行为 opt-in,既有淘汰策略不变;Phase 1 刻意不解决 shared_prefix_hash 驱动复用、HiCache/存储层元数据继承、跨进程 workflow 协调与全拓扑生产调优。

## 优化点与收益
正文未给出量化数字。正文称原型验证已存在,附元数据通路的专用单元测试(schema、API 到内部请求的传播、workflow DAG 注册与距离计算、radix 节点标注、agent_aware 淘汰顺序、省略 agent_hints 的向后兼容)。

## 现状与落地
仍在讨论/征求反馈(open)。作者就 agent_hints schema、调度器本地 DAG 跟踪是否合适的 Phase 1 边界、agent_aware 是否先以实验性淘汰策略落地、Phase 2 内容征求意见;开放问题含 TTL 应硬保护还是仅排序信号、workflow 元数据如何与 HiCache/存储后端交互、下一步加什么可观测性指标。

## 关联
- REF #21846: https://github.com/sgl-project/sglang/issues/21846

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
