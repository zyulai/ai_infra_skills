# RFC #22605: [RFC]: Separated CPU KV Cache Offloading/Transfer Process

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22605](https://github.com/vllm-project/vllm/issues/22605) |
| 作者 | ApostaC |
| 创建日期 | 2025-08-10 |
| 状态 | ⚪ closed (2026-03-27) |
| labels | RFC, stale |
| 热度 | 👍 10 · 💬 36 |
| 分类 | KV Cache / Memory |

## 一句话结论
把 CPU KV cache 卸载/收发逻辑拆到独立进程,消除 worker 内 memcpy 开销与跨 worker IPC,最大化 GPU 利用。

## 提案动机
CPU KV cache 卸载与基于 CPU 的 KV 传输对 vLLM 很重要,已有 KV-connector 方案(lmcache_connector、nixl_connector)及多个 RFC(#16144、#19854),但存在效率与复杂度问题:memcpy kernel launch 等 CPU 任务跑在 worker 进程内影响推理;KV 元数据收集需额外跨 worker IPC;传输逻辑只在 worker 被调用时触发、与推理循环耦合;卸载/逐出/CPU-based prefill 路径分散导致 h2d/d2h 重复;调度器因各种异步处理而复杂。生产落地还要求容错、CUDA graph 兼容、与混合内存分配器兼容。

## 提出的方案
- **分离到独立 "CPU KV process"**:把 KV cache offloader/sender/receiver 从 worker 进程中拆出;关键技术是 PyTorch `UntypedStorage`(原始张量数据访问)+ CUDA IPC handle(跨进程数据共享)。
- **握手**:启动时 scheduler 携 vLLM 配置向 CPU KV 进程握手,worker 握手注册各自的 GPU KV cache 张量。
- **运行时**:scheduler 经 zmq 把需要 offload/onload/send 的请求/页发给 CPU KV 进程;操作完成后回传 "finished request IDs"。
- **关键区别**:worker 不再启动任何内存相关 kernel,worker 内开销最小化、GPU 利用最大化;所有 worker 共享单一 CPU KV 内存 buffer,消除 KV 元数据的跨 worker IPC。
- 收益还包括:统一 CPU KV 传输、卸载与逐出到一个模块(无额外拷贝)、故障不影响主进程、不干扰 CUDA graph;MLA 模型 TP>1 时只需保存一份 KV cache。
- 对 layerwise pipelining 的担忧,作者回应:分离后 worker 侧开销已最小化、计算与通信天然重叠,PD 分离可用 chunk-by-chunk pipelining 达到类似效果。

## 优化点与收益
正文给出的量化数字(最坏情况:0% prefix cache hit):
- 原型(未精调)已把卸载开销从 **~10% 降到 ~3%**。
- Workload 1(Llama 3.1 8B Random,1000 input/100 output,1000 requests):Mean TPOT 162.11 ms(原始 vLLM)/ 181.27(existing connector)/ 169.16(proposed prototype);Token throughput 33161.90 / 29690.18 / 32013.08 tokens/sec。
- Workload 2(Llama 3.1 8B Random,8000 input/100 output,200 requests):Mean TPOT 200.33 / 214.42 / 209.48 ms;Token throughput 30846.83 / 28738.53 / 29833.48 tokens/sec。

## 现状与落地
已关闭(2026-03-27),labels 含 stale。Roadmap 列出 CPU KV cache server structure(#22607)、scheduler 改动(WIP,有分支代码)、CPU offloading 原型(#22608)、KV 收发原型、基于逐出的卸载等;各项完成情况正文看不出,需查关联 PR。

## 关联
- [#22607](https://github.com/vllm-project/vllm/issues/22607)(CPU KV cache server structure)
- [#22608](https://github.com/vllm-project/vllm/issues/22608)(CPU offloading 原型)
- [#16144](https://github.com/vllm-project/vllm/issues/16144)、[#19854](https://github.com/vllm-project/vllm/issues/19854)(相关 RFC)
- [lmcache_connector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py)(现有方案)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
