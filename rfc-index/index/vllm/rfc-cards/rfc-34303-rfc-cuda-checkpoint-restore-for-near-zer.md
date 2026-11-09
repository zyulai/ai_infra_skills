# RFC #34303: [RFC]: CUDA Checkpoint/Restore for Near-Zero Cold Starts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34303](https://github.com/vllm-project/vllm/issues/34303) |
| 作者 | elizabetht |
| 创建日期 | 2026-02-11 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 30 · 💬 34 |
| 分类 | Other |

## 一句话结论
用 NVIDIA CUDA checkpoint/restore API 进程级保存/恢复 GPU 状态,让 vLLM 获得近零冷启动的挂起/恢复与模型换入能力。

## 提案动机
冷启动延迟阻碍多模型共享 GPU 互换(需 30–120s 完全重启,视模型大小)、scale-to-zero serverless(每次扩容付全额冷启动,大模型做不了 sub-minute SLA)与成本摊薄。冷启动最贵的三步——权重传 GPU (~2-10s)、`torch.compile` (~5-30s)、CUDA graph capture (~2-10s)——正是 CUDA checkpoint 能完全消除的;Modal 已用该 API + vLLM 实测 10x 冷启动改进,但目前没有引擎把它做成原生内置能力。

## 提出的方案
- 基于 CUDA driver 570+ 的进程级 checkpoint API(Lock/Checkpoint/Restore/Unlock/Get State,状态机 RUNNING→LOCKED→CHECKPOINTED),配合 CRIU;保留设备内存、CUDA 内核(含 torch.compile 产物)、CUDA objects 与 CUDA graphs。
- **Tier 1(进程内 suspend/resume)**:新增 `POST /suspend`、`POST /resume`、`GET /is_suspended`(与 `/sleep` 语义不同:suspend 销毁 NCCL communicator,恢复需 `_reinit_nccl()`,TP=4/8 估计 ~1-3s);新增 `CudaCheckpointer` 包装与 `SleepMode` 枚举(`disabled`/`cumem`/`cuda_checkpoint`/`auto`)。
- **Tier 2(CRIU 全进程落盘)**:`CheckpointOrchestrator` 编排 `criu dump/restore`,提供 `vllm checkpoint`/`vllm restore` CLI 与 `--leave-running` 热快照;pre-dump 清理 `/dev/shm/sem.*`、对 >7B 模型设 `--ghost-limit 128M`。
- **Tier 3(未来)**:auto-suspend/auto-resume——空闲(默认 300s)自动挂起、新请求透明恢复(额外延迟 ~3-8s),`min_uptime_s` 默认 60s 防抖。
- 已知坑:含 NCCL 集合通信的 CUDA graph 恢复后可能仍需重捕获(开放调查项);跨 GPU 恢复需 driver 580+ 的 UUID remapping。

## 优化点与收益
正文照抄的数字(均为 RFC 估计值,条件如原文):
- Modal 用 CUDA checkpoint API + vLLM:**10x 冷启动改进**(见其博客)。
- 往返耗时对比:`cumem` sleep/wake-up 模式——进入 ~1-2s、恢复 ~5-30s、总计 **~6-32s**;`cuda_checkpoint` 模式——进入 ~2-5s、恢复 ~2-5s(含 NCCL reinit)、总计 **~4-10s**。
- CUDA graphs 与 torch.compile 产物保留(现有 sleep 模式唤醒后需重新捕获/重编译)。

## 现状与落地
仍在讨论/待实现:原帖自标 Status: Draft,scope 限 v1 引擎;按 Phase 1(核心基础设施:C 扩展、Python 包装、SleepMode 枚举)起步;与 sleep mode 修复(#32714,v0.14.0 起损坏)共享 dispatch 配置但机制独立,可并行开发。注:pack 正文在 20k 字符处截断,本卡基于现有内容成卡。

## 关联
- [#33930](https://github.com/vllm-project/vllm/issues/33930)(Issue 出处)
- [#32714](https://github.com/vllm-project/vllm/issues/32714)(sleep mode bug)
- [NVIDIA cuda-checkpoint CLI](https://github.com/NVIDIA/cuda-checkpoint)
- [Modal: GPU mem snapshots 博客](https://modal.com/blog/gpu-mem-snapshots)
- [dynamo/chrek](https://github.com/ai-dynamo/dynamo/tree/main/deploy/chrek)(CRIU + CUDA checkpoint 编排器)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
