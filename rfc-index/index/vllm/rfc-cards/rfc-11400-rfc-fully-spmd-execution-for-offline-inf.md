# RFC #11400: [RFC]: Fully SPMD Execution for Offline Inference

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11400](https://github.com/vllm-project/vllm/issues/11400) |
| 作者 | eric-haibin-lin |
| 创建日期 | 2024-12-21 |
| 状态 | ⚪ closed (关闭于 2025-01-21) |
| labels | RFC |
| 热度 | 👍 23 · 💬 1 |
| 分类 | Other |

## 一句话结论
为离线推理引入完全 SPMD 式的 LLMEngine 执行模式:每 GPU 一个独立引擎、无中心控制器,提升吞吐并利于 RL/RLHF 场景。

## 提案动机
分布式离线推理中,vLLM 采用单控制器范式(如 Ray Driver)广播调度输出、再汇集 worker 输出,`_run_workers` 带来不可忽视的派发开销;Ray DAG(#6556)与 multi-step scheduling(#6854)虽试图消除,但如 Pathways/HybridFlow 所论,单控制器的派发开销在完全 SPMD 范式下才更易根治。此外单控制器范式下 RL/RLHF 后训练(HybridEngine 式共置部署)难以实现进程合并与权重重分片,多节点离线推理还需先搭 Ray 集群,部署复杂。

## 提出的方案
- 每个 GPU 一个独立 `LLMEngine`,各含一个 vLLM scheduler 与 `SPMDGPUExecutor`(cache engine + model);无中心控制,各引擎调度自己的数据且行为保持一致
- 确定性保证:修改 `determine_num_available_blocks()` 并用 all_reduce(MIN) 同步各 worker 的块可用数据,保证 GPU/CPU cache 尺寸一致;离线场景同 DP 组各引擎同时收到相同输入,配合一致的缓存机制与 FIFO 调度,调度行为确定且相同
- 该 SPMD 版 vLLM 已在 HybridFlow(veRL)v0.6.3、v0.5.4、v0.4.2、v0.3.1 中实现,本 RFC 提议将其迁移进 vLLM LLMEngine
- 面向 RL/RLHF 的补充:SPMD 范式下的模型权重同步 API(每迭代同步)、KVCache offload(actor 训练阶段卸载 KV cache 降内存开销)
- 多节点离线推理简化为单命令:`torchrun --nnodes 2 --nproc_per_node 8 offline_inference.py`;Roadmap 还包括支持多模态/投机解码等离线推理、以及基于 SPMD 的流水线并行

## 优化点与收益
正文未给出量化数字。定性收益(正文「Major Benefits」):1) 更高的离线推理吞吐;2) RL/RLHF 训练中更快的权重重分片与更易实现;3) 更好的多节点离线推理与数据并行支持;4) [规划中] SPMD 下更易实现 LLMEngine 流水线并行。与 TensorRT-LLM 等 SPMD 执行框架对齐。

## 现状与落地
已关闭。方案在 veRL 中已有实现(v0.3.1–v0.6.3),迁移到 vLLM LLMEngine 的结果正文未写明,落地情况需查关联 PR。

## 关联
- #6556
- #6854
- veRL SPMD 实现:https://github.com/volcengine/verl/tree/main/verl/third_party/vllm
- HybridFlow 论文(Eurosys 2025)
- Pathways 论文(MLSys 2022)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
