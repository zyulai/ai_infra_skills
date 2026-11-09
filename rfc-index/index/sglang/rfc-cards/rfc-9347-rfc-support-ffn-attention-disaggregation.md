# RFC #9347: [RFC] Support FFN-Attention Disaggregation Strategy for scientific research and low capable NIC devices

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9347](https://github.com/sgl-project/sglang/issues/9347) |
| 作者 | yiakwy-xpu-ml-framework-team |
| 创建日期 | 2025-08-19 |
| 状态 | ⚪ closed (2025-11-17) |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议面向科研场景与低端 NIC 设备支持 FFN-Attention 分离 (AFD) 分片流水线,用 Dynamo graph pass 重排计算图数据流实现流水线化。

## 提案动机
芯片间通信日益重要:NIC 带宽逐代提升而 FLOPS 利用率更低、一切受网络制约;NVIDIA 与 AMD 新近的 Two Shot 通信库解锁了更复杂的流水线方案。解码阶段 attention 可能计算受限(MLA)或内存受限(如 GPT-OSS 8:1 GQA);相比 MoE Expert parallel,把 experts 分给不同卡组没有负载均衡问题,代价是通信时间。对使用昂贵路由(国产 NIC 卡)的环境,理论上难以下结论,支持该特性后可立即验证;即使未必带来即时收益,对科研也有价值。

## 提出的方案
- PoC & MVP:在 NV/AMD 卡上演示将 experts FFN 与 attention 分片到不同卡组;为连续 FFN/attention 算子创建通信 ops;以 IB 为重要评测环境,RoCE-only 方案不优先;将 FFN/Attn 调度到不同 micro batch 以流水线化整体计算
- 高效 Ping-Pong Pipeliner:FFN 与 Attn 分属两组 GPU 需流水线提升利用率,针对给定模型的流水线优化并不平凡
- 用 Dynamo graph 的 transformer/pass 重排任意计算图数据流实现 FFN-ATTN 流水线;模块拓扑序由 Constrained Topological Sorter 模块完全解析
- 映射关系:M(streams) : 1(GPU),1(Process) : N(不同角色的 GPUs)
- 详细设计在外部 Google 文档讨论;GPT-OSS 相关计算报告([1])将单独发布

## 优化点与收益
正文未给出本方案的量化收益;文中背景数字(照抄):NIC 带宽从 Ampere 到 Hopper 提升 1.5x(600 GB/s -> 900 GB/s),从 Hopper 到 Blackwell 提升 2x(900 GB/s -> 1.8 TB/s)。另注明按 [1] 的计算,以 GPT-OSS 为例 MoE parallel 比把 Attn 与 FFN 分片到不同卡组更快。

## 现状与落地
已关闭(2025-11-17)。正文为 PoC/MVP 提案与设计讨论;落地情况需查关联 PR。

## 关联
- vllm#22799(互补提案)
- 设计讨论文档(docs.google.com,见正文)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
