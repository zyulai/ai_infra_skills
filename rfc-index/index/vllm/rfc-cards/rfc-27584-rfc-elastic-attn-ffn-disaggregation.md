# RFC #27584: [RFC]: Elastic Attn-FFN Disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27584](https://github.com/vllm-project/vllm/issues/27584) |
| 作者 | wabluy |
| 创建日期 | 2025-10-27 |
| 状态 | ⚪ closed(关闭于 2026-06-04) |
| labels | RFC, stale |
| 热度 | 👍 3 · 💬 6 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议基于 Ray 的弹性 AFD 架构:服务不中断地在线调整 Attention/FFN 成员与布局,兼顾扩缩容前后吞吐。

## 提案动机
在 AFD(Attention-FFN Disaggregation)下,vLLM 的大规模 MoE 推理分布在两个执行侧:Attention 侧负责请求批处理、KV cache 管理与路由;FFN(Experts)侧执行专家 MLP。#22799 将 AFD 引入为横跨 DP-Attention 与 EP-FFN 的单实例。为进一步提升在线服务性能与可靠性,AFD 的在线弹性变得重要——在流量尖峰与节点故障时保持 QoS 与吞吐。先前的 Elastic EP(#20323)只建立了 DeepEP 的在线弹性,协调 AFD 范围更广:需要一种跨控制面与数据面联合协调 Attention 与 FFN 的新架构,并为 AFD 流水线重新定义负载均衡。RFC 有两个目标:1) 单个弹性 AFD 实例内低 MTTR——服务不停机地在线编辑 Attention/FFN 组成员与放置、调整 DP rank、调整专家放置;2) 扩缩容前后保持高吞吐——通过调整流水线分配、micro-batch 分布、线程、流与 SM 利用率等,调优 Attention 与 FFN 路径上的每节点并发。

## 提出的方案
基于 Ray 的统一弹性 AFD 实现,控制面与数据面清晰划分,并更新 AFDConnector 以支持实例内弹性通信(详细设计文档见正文 Google Docs 链接):
1. **Ray 基础弹性**:在单个 AFD 实例内在线编辑 DP-Attention 与 EP-FFN 的成员与放置(扩缩容与替换);用 Ray 做控制与执行编排,对成员与放置做最小化的原地修改,输出更新后的 actor 集合与放置,在 micro-batch 或 sequence 边界执行切换。
2. **弹性通信域统一接口**:提供单一抽象,在成员或放置变化时更新通信域;StepMesh 与 Global Communicator 作为可插拔实现,向上层屏蔽模式差异;在通信层更新 AFDConnector,使其在同一 AFD 实例内携带并更新所需元数据(成员、端点、拓扑)。
3. **里程碑**:M1 基线(Ray + 弹性通信);M2 vLLM 容错适配(#27866)+ 基于 Elastic NBO 的弹性流水线负载均衡,在 Attention 与 FFN 间调节负载;M3 异步计算图重建与捕获(可与服务并行进行,尽量复用现有产物)。

## 优化点与收益
收益为在线服务弹性与可靠性:节点故障/流量尖峰下不停机完成成员与放置调整(低 MTTR),扩缩容前后通过负载再分配保持吞吐与 QoS。正文未给出量化数字。

## 现状与落地
已关闭(2026-06-04)。正文为带设计文档与三阶段里程碑的提案(反馈期未填);里程碑是否完成、方案是否合入从正文无法看出,落地情况需查关联 PR。

## 关联
- [AFD RFC #22799](https://github.com/vllm-project/vllm/issues/22799)(AFD 引入提案)
- [AFD PR #25162](https://github.com/vllm-project/vllm/pull/25162)
- [Elastic EP #20323](https://github.com/vllm-project/vllm/issues/20323)(DeepEP 在线弹性先例)
- [#27866](https://github.com/vllm-project/vllm/issues/27866)(vLLM 容错适配,Milestone 2 关联)
- [设计文档(Google Docs)](https://docs.google.com/document/d/1_bT3aMlNUWBlrirddWquzij3oxQTBhwW7nsBik4FlOY/edit?tab=t.0)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
