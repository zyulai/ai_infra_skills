# RFC #10727: [RFC]: Implement disaggregated prefilling using Mooncake

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#10727](https://github.com/vllm-project/vllm/issues/10727) |
| 作者 | alogfans |
| 创建日期 | 2024-11-28 |
| 状态 | ⚪ closed(关闭于 2025-03-28) |
| labels | RFC, stale |
| 热度 | 👍 6 · 💬 2 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议把 vLLM 与 Mooncake Transfer Engine 集成实现 disaggregated prefilling:数据面先用 Transfer Engine 替换 NCCL,后续以 Mooncake Store 支撑 M prefill + N decode。

## 提案动机
分离式 prefilling/decoding 有望在 LLM 推理(如长文档场景)取得更好性能;issue #5557 已提出良好范式。与此同时,Mooncake——一个以 KVCache 为中心的 LLM serving 分离架构——的 Transfer Engine 已开源。与 NCCL 相比它有三个特性:统一编程接口,覆盖 DRAM-to-DRAM(本地与远程)、DRAM-to-GPU VRAM(本地与远程)、DRAM-to-远程 NVMe 设备的数据传输;支持 TCP、RDMA、NVMe-of 协议;具备拓扑感知的路径选择,可聚合多块 NIC 的带宽。

## 提出的方案
- 将 vLLM 与 Mooncake 集成,作为分离式 prefilling 的实现路径。
- 第一步:已实现原型,在数据面用 Mooncake Transfer Engine 替换 NCCL,原型开放给社区试用并征求意见。
- 第二步(规划):开发 Mooncake Store,完整支持分离式 prefilling(M 个 prefill 节点 + N 个 decode 节点)并达到生产可用。
- Mooncake 整体架构见其仓库 architecture 文档。

## 优化点与收益
正文未给出量化数字。收益点来自 Transfer Engine 相对 NCCL 的能力差异:统一接口覆盖 DRAM/VRAM/NVMe 多种传输形态,支持 TCP/RDMA/NVMe-of 多协议,拓扑感知路径选择聚合多 NIC 带宽;目标是让长文档等场景从 prefill/decode 分离中获得性能收益。

## 现状与落地
已关闭(2025-03-28)。正文提到数据面替换 NCCL 的原型已完成、Mooncake Store 为规划中的后续工作;vLLM 侧最终落地情况需查关联 PR(vLLM 后续另有独立的 KV connector / disaggregation 实现,与本 RFC 的承接关系需另行核实)。

## 关联
- 关联 issue #5557: https://github.com/vllm-project/vllm/issues/5557
- Mooncake: https://github.com/kvcache-ai/mooncake
- Mooncake 架构文档: https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/architecture.md

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
