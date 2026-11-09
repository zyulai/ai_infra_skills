# RFC #4461: [RFC]: Initial support for Pipeline Paralleism

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#4461](https://github.com/vllm-project/vllm/issues/4461) |
| 作者 | andoorve |
| 创建日期 | 2024-04-29 |
| 状态 | ⚪ closed (关闭于 2024-07-02) |
| labels | RFC |
| 热度 | 👍 21 · 💬 1 |
| 分类 | Feature / Ecosystem |

## 一句话结论
为 vLLM 引入流水线并行初始支持:按层切分模型到多设备,用「虚拟引擎」多流并发消除设备空闲。

## 提案动机
流水线并行把模型层切分到多个设备(如 12 层模型分给 4 设备各 3 层),各设备执行完自己的部分后把数据传给下一设备,以流水线方式处理微批次。相比张量并行,它的通信开销更低;两者并不互斥(如节点内 TP、通信昂贵的节点间 PP)。这对需要多机运行的超大模型,以及高通信开销(如无 NVLink)的机器很重要。

## 提出的方案
- 当前阶段求最简实现(简单优先于优化),范围限定在 `AsyncLLMEngine` 与 `RayGPUExecutor`
- 引入「虚拟引擎」(virtual engines):数据流数量等于流水线阶段数,每个虚拟引擎拥有各自的 scheduler(含 block manager)与 cache engine,调度完全独立;新请求按代价函数分配到各 scheduler
- 模型侧:每设备只保留 num_layers/pp_world_size 层,非首末阶段插入相应的 send/recv
- 引擎侧:`step_async`/`engine_step` 增加 `virtual_engine` 参数;运行循环允许每个虚拟引擎同时 1 个 async step;`LLMEngine` 持多个 scheduler,按 sequence 数最少者分配
- 缓存:每个虚拟引擎按 num_blocks/pp_stages 均分 cache 空间
- 执行器:采样改在最后一个 PP stage 的第一个 TP worker 上进行(每个 TP 组的第一个 worker 为该组 driver);广播的 src 改为各 TP 组第一个 worker

## 优化点与收益
正文未给出量化数字。定性收益:相比张量并行通信开销更低;单流时 worker 有 3/4 时间空闲等待,增加与阶段数相同的数据流(4 个)即可让每个 worker 饱和。

## 现状与落地
已关闭。正文给出初始 PR 与 tracking items(#4412),实现以该 PR 推进,落地情况需查关联 PR。

## 关联
- 初始 PR(含 tracking items):https://github.com/vllm-project/vllm/pull/4412

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
