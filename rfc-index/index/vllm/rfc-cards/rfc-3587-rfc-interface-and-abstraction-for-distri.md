# RFC #3587: [RFC]: Interface and Abstraction for Distributed Inference Environment

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3587](https://github.com/vllm-project/vllm/issues/3587) |
| 作者 | youkaichao |
| 创建日期 | 2024-03-23 |
| 状态 | ⚪ closed (关闭于 2024-06-14) |
| labels | RFC |
| 热度 | 👍 10 · 💬 18 |
| 分类 | Other |

## 一句话结论
为分布式推理环境提出 Launcher/Coordinator/Communicator 三层抽象与接口,补齐缺失的 Coordinator 层以消除死锁与 hack。

## 提案动机
当前 vllm 的分布式推理环境相当混乱,经常出现死锁与挂起(如 #3455、#2770、#3559)。升级 PyTorch 2.2.0 时问题凸显:NCCL 从 2.18.1 升到 2.19.3,cudagraph capture 内存从每图 10MB 涨到 100MB,几十个 cudagraph 加起来数 GB,导致 vllm 出错。作者直言:分布式推理在当前代码库中是个头痛问题,"如果它能跑,万岁;跑不了也别惊讶"。

## 提出的方案
- 三层抽象:Launcher(启动进程,可用 ray,单机也可用 Python 原生 multiprocessing)、Coordinator(协调共享资源、广播消息;目前缺失,靠 filelock/TCP/MPI 等 ad-hoc hack 实现)、Communicator(跨设备大 tensor 通信,如 nccl,与厂商相关)
- Launcher→Coordinator 接口:launch_id、world_size、local_world_size、rank、local_rank、master_addr、master_port 等;launch_id 可作"随机种子"推导 master_port
- Coordinator 接口:is_master()、is_local_master()、broadcast(bytes, src)、barrier()
- Communicator 接口:参照 nccl 设计,master 获取并广播唯一 token,各进程据此初始化;至少提供 in-place allreduce
- Coordinator 参考实现:torch.distributed 的 gloo 后端(CPU tensor),不引入新依赖;大 tensor 通信直接调厂商库,绕开 torch.distributed.all_reduce 的复杂行为(可能初始化 autograd、破坏 cudagraph)

## 优化点与收益
正文未给出量化数字。定性收益:系统始终处于分布式环境(只是 world_size 不同),易扩展多机;硬件厂商只需提供 torch.Tensor 集成与一个 .so 通信库,快速接入甚至只需实现 allreduce 一个 op。

## 现状与落地
已关闭。正文表示征集讨论一周(至 2024-03-31)后即开始实际重构代码,正文未写明最终结果,落地情况需查关联 PR。

## 关联
- https://github.com/vllm-project/vllm/issues/3455
- https://github.com/vllm-project/vllm/issues/2770
- https://github.com/vllm-project/vllm/issues/3559
- https://github.com/vllm-project/vllm/pull/3442

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
