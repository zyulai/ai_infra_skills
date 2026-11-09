# RFC #6854: [RFC]: Multi-Step Scheduling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6854](https://github.com/vllm-project/vllm/issues/6854) |
| 作者 | SolitaryThinker |
| 创建日期 | 2024-07-26 |
| 状态 | ⚪ closed (关闭于 2024-12-08) |
| labels | RFC, stale |
| 热度 | 👍 28 · 💬 16 |
| 分类 | Overlap / Scheduler |

## 一句话结论
一次调度执行多步 decode(MS-8/MS-16),把每步的 CPU 调度与同步开销摊薄,消除 GPU 空泡。

## 提案动机
每个 decode batch 都有很高的 CPU 开销:sampled token 从 GPU 传回 CPU 做去 token 化与响应、为用户生成输出(Python 化张量)、CPU 准备下一步输入元数据、vLLM scheduler 调度。结果是 GPU 常空闲等待 CPU(5-13ms 的 GPU bubble)。多步调度指执行多次 decode 后才做 GPU-CPU 同步并调用调度器;sampled token 的 GPU->CPU 传输也改为在单独 CUDA stream 中异步进行,CPU 领先 GPU 运行时开销几乎为零。

## 提出的方案
- 调度器侧:固定 n 步并额外分配 blocks;仅用于 decode,prefill 照旧;为 scheduler config/engine 参数/CLI 增加返回 lookahead slots 的开关;未用完 lookahead slots 时跳过 vLLM scheduler 调用
- worker 侧:sampler 不做 CPU<->GPU 同步,只在 GPU 内生成 next token;每迭代把 tokens 广播到所有 worker;用 CUDA kernels(而非过慢的 Torch)就地更新下一步输入;异步传输 sampled tokens 到 CPU
- 扩展 `ExecuteModelRequest` 与 `RequestOutput/SamplerOutput` 携带 multi-step 状态;跟踪当前步数、剩余 lookahead slots、留在 GPU 的 `sampled_token_ids`、`sampler_output_ready_event` 等 CUDA 事件
- `ModelRunner` 改动:TP/PP 下广播 sampled token 供各 rank 调 `advance_step`;用 CudaEvents 与前序 forward 同步避免 CPU 破坏在用 GPU 张量;CPU 领先时即时 Python 化就绪张量;lookahead slots 用尽时阻塞等待
- 原型:基于投机解码 `T1DraftModelRunner` 逻辑构建非投机版 MultiStepModelRunner(`workers/` 下),目标合并前泛化进现有 ModelRunner;PP+multi-step 需特殊处理否则各步间产生空泡

## 优化点与收益
正文数字(照抄):
- GPU bubble:5-13ms(torch profile 中每个红框约 4ms;baseline 每次 decode 都产生,MS-8 每 8 次 decode 只需一次)
- 吞吐(ShareGPT,benchmark_serving.py,无限请求速率,1k requests,开 CUDA graph,无 chunked prefill,输入/输出长度取自 ShareGPT 数据集):

| Single GPU | Baseline (Req/s) | MS-8 (Req/s) | MS-16 (Req/s) |
| --- | --- | --- | --- |
| A10G 8B Llama | 5.20 | 5.89 | - |
| H100 8B Llama | 20.66 | 40.06 | 43.31 |
| H100 30B Llama | 9.23 | 13.09 | 13.23 |

## 现状与落地
已关闭。正文含原型设计与 benchmark,未写明最终实现结论,落地情况需查关联 PR。

## 关联
- PP+multi-step 设计文档:https://docs.google.com/document/d/1_qb_HKZj7cpwsCTQ4JmxHZVUxtRcpgMCjWqHR7JMPfw/edit?usp=sharing

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
