# RFC #49765: [RFC]: Native Disaggregated Pull-Based Queue Worker Interface and Heuristic Pull Router

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49765](https://github.com/vllm-project/vllm/issues/49765) |
| 作者 | aojea |
| 创建日期 | 2026-07-24 |
| 状态 | ⚪ closed(关闭于 2026-08-08) |
| labels | RFC |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Disaggregated Serving |

## 一句话结论
新增原生"拉取式"队列 Worker 接口与启发式 Pull Router:worker 在有空闲 GPU 槽与 KV 块时主动拉任务,取代 push 代理。

## 提案动机
传统 push 型 HTTP 代理把请求推向 vLLM worker:在异构 GPU 集群(如 80GB A100 与 24GB RTX 4090 混布)上大上下文 prompt 路由不当易 OOM;缺拓扑感知造成跨区/跨云流量成本与延迟尖刺;用活跃连接数估算负载,突发流量压垮 KV 分配导致 OOM。Pull 模型下 worker 只在 GPU 执行槽与 KV cache blocks 可用时,经 credit grants(`TaskGrantRequest`)向外建立 gRPC 流式连接拉取任务,天然形成背压、可按硬件能力过滤任务。

## 提出的方案
- 新增 `vllm.entrypoints.pull_worker.worker` 接口,纯增量(additive),不改任何核心引擎文件与既有 HTTP push 端点。
- 经 gRPC `queue.proto` 扩展:任意语言实现该 proto 即可写自定义队列实现。
- 可插拔遥测:`WorkerStatus` 分 Core Metrics(num_running_reqs、free_gpu_kv_blocks 等安全数据)与 Extended Metrics(经自定义 StatLoggerBase 插件把 K8s zone 标签、活跃 LoRA 适配器等注入 custom_metrics),路由器不进 vLLM 核心即可执行复杂业务逻辑。
- Worker 上报活跃 radix tree 块哈希(`active_prefix_hashes`),中央路由器把带相同 system prompt 前缀的任务派给已"热"的 worker,消除 prefill 开销。
- 无需 GPU worker 前置 NGINX/Envoy 负载均衡器;参考实现放 `examples/heuristic_pull_router/`(无状态 gRPC server、Redis 后端、可插拔 BasePolicy 启发式)。

## 优化点与收益
基准:`benchmark_bakeoff --num-prompts 100 --request-rate 25.0`(单 worker 与 2-worker 两场景;pack 未注明硬件与模型)。2-worker 场景:P99 TTFT `95.86 ms` → `39.20 ms`(-54.2% Latency),P50 TTFT `76.65 ms` → `27.95 ms`(-63.5%),平均端到端延迟 `2855.00 ms` → `1861.59 ms`(-34.3% Faster Completion),前缀缓存命中率 `30.0%` → `96.0%`(+66.0% Cache Reuse);Redis 后端命令延迟均值 `1.589 ms`、P99 `1.686 ms`,吞吐 `2758.8 tok/s` 持平("Zero Throughput Penalty")。单 worker 场景:P99 TTFT `93.66 ms` → `39.20 ms`(-53.0%),命中率同为 `30.0%` → `96.0%`。

## 现状与落地
已关闭 —— 正文处于提案阶段,作者以新人身份征求架构意见,提议两周反馈期(目标 2026-08-07),另维护 Golang+Kubernetes 的私有原型并表示愿在 vllm-project 伞下维护官方参考实现;正文未写最终是否合入,落地情况需查关联 PR。

## 关联
- [参考实现分支 aojea/vllm `vllm_pull`](https://github.com/aojea/vllm/tree/vllm_pull)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
