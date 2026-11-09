# RFC #22799: [RFC]: ATTN-FFN Disaggregation for MoE Models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22799](https://github.com/vllm-project/vllm/issues/22799) |
| 作者 | chopper0126 |
| 创建日期 | 2025-08-13 |
| 状态 | ⚪ closed(关闭于 2026-05-31) |
| labels | RFC, stale |
| 热度 | 👍 9 · 💬 17 |
| 分类 | MoE |

## 一句话结论
提议在 vLLM 引入 Attention–FFN 分离(AFD)让注意力与 MoE 独立伸缩,并提供 AFConnector 自定义通信接口。

## 提案动机
MoE 解码阶段的稀疏性收益推动专家并行(EP)持续扩张;既有设计(如 DeepEP)把 EP 分片放在数据并行(DP)rank 上,规模化时扩展性受限。ByteDance(MegaScale-Infer)、StepFun(Step-3)、Huawei(xDeepServe)已先后提出 Attention–FFN 分离(AFD):注意力阶段内存受限、FFN/专家阶段计算受限,单一同构部署无法同时优化两者,模块级异构放置随 EP 扩展变得有益。本 RFC 提议在 vLLM 引入 AFD:资源级解耦、独立伸缩,并在「完全均衡」的流水线中重叠通信提升吞吐;首版为 eager-mode MVP、无自定义内核,注意力侧跑多个 DP 副本、MoE 侧跑 EP 副本,支持跨节点 M2N 路由并保证正确性,后续再提供稳定的通信/负载均衡/弹性接口。

## 提出的方案
- 核心变更一 AFD on vLLM(MVP):面向 DeepSeek-v3 等 MoE 模型,eager 模式 + torch 通信 API 做 M2N。路线图已勾选:模型侧重定义模型结构/权重加载/模块图;服务侧定义全局 topo、经 torchrun 启动独立 MoE 服务(暂以 P2P 与注意力节点通信);输出正确性验证。待办:可配置 micro-batch 的流水线与异步 M2N 通信、MTP 等性能优化与自定义负载均衡策略、支持自定义 MoE 模型并完善 AFConnector 代码
- 核心变更二 AFConnector 接口(面向 XPU):支持自定义 AF 通信后端(如 stepmesh)。已勾选:收集注意力/FFN 实例 IP 与 rank 并初始化通信资源;待办:tensor 级异步 send/recv 与 all-gather/reduce-scatter API、硬件无关接口、Python setuptools entrypoints 插件机制
- 后续特性:图模式支持(XPU 上的 AFD 加速)、AF+PD 全分离、统一基准测试框架、弹性容错 AFD(在线扩缩容保 KV 一致性与正确路由)、服务管理(对接 AIBrix)

## 优化点与收益
正文未给出量化数字(动机层面:注意力与 FFN 资源画像互补,异构放置 + 流水线通信重叠以提升吞吐)。

## 现状与落地
已关闭(2026-05-31,label 含 stale)。两项核心变更路线图仅部分勾选(分离部署、正确性验证、通信资源初始化完成;流水线、性能优化、接口完善未勾选);最终落地情况需查两份设计文档与关联 issue #21644。

## 关联
- DeepEP:https://github.com/deepseek-ai/DeepEP
- MegaScale-Infer:arXiv:2504.02263
- Step-3:arXiv:2507.19427
- xDeepServe:arXiv:2508.02520
- 关联 issue:https://github.com/vllm-project/vllm/issues/21644

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
