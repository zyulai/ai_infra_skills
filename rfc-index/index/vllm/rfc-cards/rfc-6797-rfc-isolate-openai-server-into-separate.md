# RFC #6797: [RFC]: Isolate OpenAI Server Into Separate Process

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6797](https://github.com/vllm-project/vllm/issues/6797) |
| 作者 | robertgshaw2-redhat |
| 创建日期 | 2024-07-25 |
| 状态 | ⚪ closed (2024-12-16) |
| labels | RFC |
| 热度 | 👍 10 · 💬 27 |
| 分类 | Frontend / Serving / API |

## 一句话结论
提议把 OpenAI API server 与 AsyncLLMEngine 拆成两个进程、经 gRPC/protobuf 通信,让 CPU 侧工作与 GPU 执行更好重叠。

## 提案动机
目前 OpenAI API server 与 AsyncLLMEngine 共享同一个 asyncio 事件循环,API server 与引擎的 CPU 部分互相争抢同一批资源。作者给出 Llama-3-8B 跑 ShareGPT、QPS=10、H100 上的时间分布图:时间分为三块——重叠的 GPU 执行时间(`execute_model` 调用)、LLM 引擎的 CPU 执行时间、其余即 API server 时间;由于 `execute_model` 经 `run_in_executor` 执行、不阻塞事件循环,部分 GPU 时间其实已与 API server 重叠,进一步重叠的空间在进程隔离。

## 提出的方案
- 通过多进程把 API server 与 AsyncLLMEngine 拆分为两个进程,以 gRPC + protobuf 通信,实现两者更好的重叠;这大致是 TGI 采用的架构(但本提案包含的服务项更少);
- 初始目标:编写 protobuf API 大致对齐 `AsyncLLMEngine.generate()` 方法,把 API server 拆到独立进程并测量加速;
  - 排除 LogitsProcessors(已共识其不属于 SamplingParams),改用对应的外部 API 参数(如 guided decoding);
  - 响应可用 gRPC streaming,与 generate 返回的流式结果自然对应;cancellation 映射为 request abort;
- 后续方向:把目前跑在 `AsyncLLMEngine` 内的事项移到 API server 进程以更好地与 GPU 重叠——
  - 引擎支持纯 token-id 进/出,使调用方可完全不做事后 tokenization/detokenization(当前有些位置无条件做,如为返回结果 detokenize prompt);
  - 把 LogitsProcessor 的构造从 OpenAI API 代码中解耦为独立工具层,由 gRPC API 层调用以组装完整 SamplingParams;
  - 解耦增量 detokenization 逻辑,使其也能在 API server 进程中使用;
  - API server 进程可直接拦截/处理 stop 字符串,按需中止请求。

## 优化点与收益
收益是让 API server 的 CPU 工作与引擎 CPU/GPU 执行在进程间重叠,减少事件循环争抢带来的等待;作者计划在拆分后实测加速。正文未给出量化数字(性能数据仅存在于正文引用的示意图中)。

## 现状与落地
已关闭(2024-12-16)。正文处于方案提议阶段(设定初始目标与后续方向,CC 多位维护者),正文未记录方案是否被实现或被替代,落地情况需查关联 PR。

## 关联
- https://github.com/vllm-project/vllm/blob/d92b3c5cdef59533347ac714a70274f186943019/vllm/executor/gpu_executor.py#L143 (`execute_model` 经 run_in_executor 执行的位置)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
