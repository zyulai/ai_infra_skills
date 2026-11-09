# RFC #31848: [RFC]: Native Weight Syncing APIs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31848](https://github.com/vllm-project/vllm/issues/31848) |
| 作者 | hao-aaron |
| 创建日期 | 2026-01-07 |
| 状态 | 🟢 open |
| labels | RFC, quantization |
| 热度 | 👍 14 · 💬 11 |
| 分类 | Other |

## 一句话结论
为 RL/post-training 场景提供原生、标准化的在线权重同步 API,免去各框架重复造轮子。

## 提案动机
vLLM 不只是面向终端用户的推理运行时,也承担 post-training 中的请求服务:训练期间必须周期性同步模型权重,使 rollout 与基础模型权重的分布一致。当前官方做法是在每个 worker 类中注入额外功能来创建进程组并广播(见 rlhf.py 示例),没有标准化的在线权重同步方法。SkyRL、VeRL、TRL 等开源项目各自实现权重同步基础设施,给采用 vLLM 做 post-training 推理后端的开发者带来额外复杂度:重复劳动、被特定实现版本锁定、被特定 torch.distributed 后端锁定。

## 提出的方案
- **原生支持权重同步**:同时覆盖 offline 推理(LLM 与 AsyncLLM)和 OpenAI 兼容 API 的 HTTP 调用,并有原型实现(ahao-anyscale 的 weight_transfer_prototype 分支)。
- **三个新端点**:`POST /init_weight_transfer` 在 trainer(rank 0)与所有 vLLM worker 间建立进程组(参数 master_address、master_port、rank_offset、world_size;worker 计算 `rank = (dp_rank * world_size_per_dp + tp_rank) + rank_offset`,创建 `StatelessProcessGroup` 并初始化 `PyNcclCommunicator`);`POST /update_weights` 按 names/dtype_names/shapes 增量接收权重并立即 `model.load_weights`,用完即释放临时张量以避免内存累积与 OOM;`POST /finalize_weight_update` 做量化/自定义 kernel 格式的后处理——正文 EDIT 注明该端点已被 #30359 取代,不会出现在最终版本。
- **可插拔架构**:引入 `WeightTransferEngine` 抽象,把权重传输的 transport 逻辑与 worker 实现分离;支持 NCCL(broadcast)与 IPC(同节点 CUDA IPC 共享内存句柄)两种后端。
- **配置与用法**:通过 `WeightTransferConfig(backend="nccl")` 配置;正文给出 Ray actor 形式的 RLHF 完整示例(trainer 侧同步广播、vLLM 侧异步接收)。
- **进展(正文 checklist)**:PR1 新 API/NCCL/示例/量化重载集成(PR #31943)与 PR2 IPC(PR #34171)已完成;Stateful Trainer send 抽象已完成并接入 NCCL/IPC;NCCL M2N sharding-aware transfer(#46439,支持 trainer TP ≠ inference TP)已完成;WPI zero-copy weight transfer(PR #40828)与 RDT weight transfer engine 等仍待完成。
- **正确性**:由单独的 RFC #48312 跟踪,含 5 个 open weight reload bug + 5 个 KV cache invalidation bug。

## 优化点与收益
正文中唯一的量化数字:WPI zero-copy weight transfer(PR #40828)基于 Kubernetes-native 共享内存,「~20 GB/s measured」(照抄,测试条件正文未详述)。其余收益为工程层面:标准化 API 消除 SkyRL/VeRL/TRL 等项目的重复实现与版本/后端锁定。

## 现状与落地
活跃实现中:状态 open,PR1/PR2 及多项后续已落地合并,PR3 follow-ups、zero-copy 传输等仍在进行;Feedback Period 为 1-2 周。

## 关联
- PR [#31943](https://github.com/vllm-project/vllm/pull/31943)(新 API + NCCL)
- PR [#34171](https://github.com/vllm-project/vllm/pull/34171)(IPC 后端)
- [#30359](https://github.com/vllm-project/vllm/issues/30359)(Online quantization with model_loader,取代 finalize 端点)
- [#48312](https://github.com/vllm-project/vllm/issues/48312)(正确性跟踪 RFC)
- [#46439](https://github.com/vllm-project/vllm/issues/46439)(NCCL M2N sharding-aware transfer)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
