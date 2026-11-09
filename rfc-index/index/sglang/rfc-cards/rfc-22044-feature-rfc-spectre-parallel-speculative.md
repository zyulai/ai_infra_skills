# RFC #22044: [Feature][RFC] SPECTRE: Parallel Speculative Decoding with a Multi-Tenant Remote Drafter

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22044](https://github.com/sgl-project/sglang/issues/22044) |
| 作者 | zhangfeiyu5610 |
| 创建日期 | 2026-04-03 |
| 状态 | ⚪ closed (2026-07-12) |
| labels | (none) |
| 热度 | 👍 13 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
提出 Drafter/Verifier 解耦为独立异步服务的多租户远程投机解码架构 SPECTRE,跨轮重叠隐藏 draft 延迟并提高 GPU 利用率。

## 提案动机
传统投机解码中 Drafter 与 Verifier 紧耦合,严格的同步「draft-then-verify」循环导致顺序延迟高、GPU 利用率低;为 Drafter 单独占用一台 GPU 又利用率不足。SPECTRE 受 Pearl、SSD 等并行投机解码工作启发,把投机解码改造成分布式、异步协调的管线。

## 提出的方案
- Drafter 与 Verifier 均为标准 SGLang HTTP 服务,经 ZMQ(C++ pybind11 扩展 + msgpack)异步通信,协议为 `SpectreRequest`,含 DRAFT / FINISH / ABORT / REJECT 四种动作与请求轮次计数
- Cross-Round Parallel Overlap:乐观并发——假设 draft 高接受率,Verifier 校验第 k 轮时 Drafter 同时为第 k+1 轮生成 token,完全重叠两模型前向以隐藏 drafting 延迟
- Drafter 双模式共存:远端 draft 请求与自身用户流量混入同一 continuous batch;提供 Mixed(默认,最大化利用率)与 Draft-Priority(`--spectre-draft-priority`,每迭代先跑限定步数的专用 draft batch)两种调度策略
- Verifier 端 `DraftCircuitBreaker` 熔断(CLOSED / OPEN / HALF_OPEN),远端 drafting 不健康时退化为单 token 自回归解码,而不是放大延迟或死锁
- 基于 SGLang 0.5.10rc0 实现,新代码集中在 `python/sglang/srt/speculative/spectre/`,集成点为 server_args / spec_info / scheduler / cuda_graph_runner;同机 `127.0.0.1` 自动选 IPC、跨机用 TCP,Verifier 绑 Router、Drafter 连 Dealer 以便扩展一Verifier多Drafter拓扑

## 优化点与收益
Dedicated traffic 条件:bench_serving random,input-len 4000 / output-len 1000,request-rate 8,max-concurrency 32,128 prompts;Target `Qwen3-32B`、Draft `Qwen3-0.6B`、Eagle3 用 `Zhi-Create-Qwen3-32B-Eagle3`:
- Autodecode 715.97 tok/s(1.00x);Eagle3 925.88(1.29x);Standalone 992.00(1.38x);Spectre 1258.82 tok/s(**1.76x**)
Mixed traffic(Drafter 侧另跑 input-len 1000 / output-len 1000、request-rate 32、concurrency 8):Spectre-verifier 1244.71 tok/s(1.74x);Drafter 自身吞吐 Origin 3291.44 → Spectre 2869.87(**0.87x**)——系统级收益以 Drafter 峰值吞吐下降为代价。

## 现状与落地
已关闭(2026-07-12)。正文含完整实现(目录结构、CLI、复现命令)与性能评测,属已有完整原型;是否合入上游需查关联 PR。

## 关联
- Pearl: arXiv:2408.11850
- SSD(Speculative Speculative Decoding): arXiv:2603.03251
- SPECTRE paper: arXiv:2605.08151

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
