# RFC #61787: [RFC]: Layerwise KV Pooling and P/D Disaggregation for High-Hit-Rate Agent Workloads

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61787](https://github.com/vllm-project/vllm/issues/61787) |
| 作者 | weijinqian0 |
| 创建日期 | 2026-10-15 |
| 状态 | 🟢 open |
| labels | kv-connector, pooling |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Disaggregated Serving |

## 一句话结论
为 KV 命中率 >90% 的 Agent 场景提出逐层协同的 KV 池化 + P/D 分离设计:块按层回收、加载与计算流水重叠。

## 提案动机
一次 140K 输入 tokens、95% 缓存命中率的测试中,24 并发请求的 token 吞吐比单请求差约 50% —— 缓存命中并不会自动转化为高并发下的性能。两个目标:①提升 HBM 利用率,不再需要的缓存块若仍被逻辑占用,会挤占新请求的并发;②隐藏缓存加载延迟,未被覆盖的加载时间留在关键路径上,降低吞吐或增加延迟。高命中时 Prefill 主要受 HBM 容量约束;异步调度在非逐层执行下还会加剧块滞留。只改 pooling 或只改 P/D 为逐层都不够 —— 没有匹配的 handoff 粒度,Decode 无法确认某层已消费完并释放对应块,因此本 RFC 把逐层 KV 池化与逐层 P/D 分离作为协同设计提出。

## 提出的方案
- 两个协作模块共享逐层生命周期:Layerwise Pooling Module(按层组织分组 KV 块,消费完成后立即归还池中供后续组复用,防止块在传输/计算仍依赖时被复用)与 Layerwise P/D Disaggregation Module(提供 per-layer ready/consumed 边界,支撑后续组加载与当前组计算的流水;关联 PR #62504)。
- 模块交互:P/D 模块取 KV 组推进当前层计算 → Decode 消费完通知 pooling 模块到达回收边界 → pooling 回收块、后续组继续,形成加载-计算流水。
- 设计原则:pooling 单元、P/D 传输单元、消费确认与回收单元对齐到一致粒度;显式 ready/consumed 依赖控制可见性与复用;组大小平衡管理开销、传输效率与回收机会。
- 配套优化:降低每层准备开销、避免冗余 cache-key 查找、处理 H2D 拷贝与 AlltoAllV 的资源争用。
- 详细 API、状态机与实现机制均为"To be specified";附评估计划(命中率 90%/95% 与低命中点、并发 1~24、140K 输入、TTFT/TPOT/HBM 峰值等指标与消融)。

## 优化点与收益
报告数字(工作负载:Agent、命中率 >90%;逐层协同设计 vs 非逐层):TTFT 约好 `40%`,聚合吞吐约高 `20%`。正文同时声明复现所需条件(模型、硬件、命中率定义、请求集、并发、测量方法、run 数)未完整给出;更早材料曾报告 95% 命中率下约 `3%` 回归(中间结果,指标与基线未注明,不应与上述结果直接比较)。另一观察:140K tokens、95% 命中、24 并发比单请求差约 `50%`(吞吐还是延迟存疑,列为 open question)。

## 现状与落地
仍在讨论/等待评审:open 状态、0 评论;RFC 只确立架构边界与职责,组粒度、池元数据、handoff 接口、ready/consumed 信号、跨进程同步等细节均待后续设计,文末列出 7 个 open questions。

## 关联
- [PR #62504(layerwise P/D 相关工作)](https://github.com/vllm-project/vllm/pull/62504)
- [RFC #60523(Improving MooncakeConnector,含 layerwise/pooling 集成路线)](https://github.com/vllm-project/vllm/issues/60523)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
