# RFC #35848: [RFC]: Revamp Ray Distributed Executor Backend (from Ray team)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35848](https://github.com/vllm-project/vllm/issues/35848) |
| 作者 | jeffreywang88 |
| 创建日期 | 2026-03-03 |
| 状态 | ⚪ closed (关闭于 2026-08-29) |
| labels | RFC, stale |
| 热度 | 👍 10 · 💬 12 |
| 分类 | Other |

## 一句话结论
用 RayExecutorV2 替换基于 Compiled Graph 的 Ray 执行器:Ray 只做进程启动与 placement,控制面/数据面通信完全对齐 MultiprocExecutor。

## 提案动机
(1) Compiled Graph 的优化不再是主要瓶颈:async scheduling(v0.15.0)已把控制面调度与 GPU 执行重叠,torch.distributed 原生提供 NCCL collectives;(2) 维护负担:NCCL hangs 与乱序投递等稳定性问题跨多个 Ray 版本未解决,Ray Core 团队资源有限;(3) 与 MultiprocExecutor 黄金路径分叉,每个新功能都要为第二个后端单独适配,贡献者负担重;(4) Ray 的核心价值是集群编排(进程放置、故障检测、资源管理),应只用它做进程启动器与 placement manager。保留 Ray(而非整体移除)的理由:跨节点并行配置更简单、进程清理(避免 mp 后端的僵尸进程)、RL 应用需 Ray RDT 编排(与 Compiled Graph 不兼容)。

## 提出的方案
- 新 RayExecutorV2:worker 为普通 Ray remote actor;控制面用 ZMQ/SHM MessageQueue,数据面用 NCCL,均与 MultiprocExecutor 一致;热路径无 ray.remote 调用(collective_rpc 走 MessageQueue)
- API 选定 Option 1:RayExecutorV2 继承 MultiprocExecutor 以最大化复用代码(零重复;风险靠充分测试缓解;Option 2 独立实现、Option 3 抽公共 MessageQueueExecutor 均未选)
- Placement group:复用 initialize_ray_cluster(),支持客户端传入 placement group;bundle 排序(driver 节点最低 rank、同节点连续 rank)后直接定 rank,不再需要 adjust_rank RPC
- RayWorkerProc:run() 经 actor.run.remote() 启动 busy loop;就绪用 wait_for_init()+ray.get();关停用 ray.kill(),无需信号处理;driver 死亡时 Ray GCS 自动杀死 owned actors
- 迁移三阶段:环境变量 VLLM_USE_RAY_V2_EXECUTOR_BACKEND 默认 False → v0.x+2 默认 True → v0.x+4 移除旧 ray_executor.py
- 非目标:不整体移除 Ray;Compiled Graph 的通信-计算重叠不在本期范围

## 优化点与收益
正文未给出量化数字。benchmark 目标为「No overhead over MultiprocExecutor」;定性收益:多后端功能一致、跨节点 PP/TP 更简单、支持细粒度 placement(如 SkyRL 用 VLLM_BUNDLE_INDICES 做 collocation 与资源感知调度)。

## 现状与落地
已关闭。正文给出三阶段迁移计划与测试计划,落地情况需查关联 PR。

## 关联
- https://github.com/vllm-project/vllm/issues/35403
- https://github.com/ray-project/ray/issues/57696
- https://github.com/ray-project/ray/issues/59913
- SkyRL 用法示例:https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py#L144

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
