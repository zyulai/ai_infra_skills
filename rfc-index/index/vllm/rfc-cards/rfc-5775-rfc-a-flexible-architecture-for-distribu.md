# RFC #5775: [RFC]: A Flexible Architecture for Distributed Inference

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5775](https://github.com/vllm-project/vllm/issues/5775) |
| 作者 | youkaichao |
| 创建日期 | 2024-06-24 |
| 状态 | ⚪ closed (关闭于 2024-06-26) |
| labels | RFC |
| 热度 | 👍 13 · 💬 7 |
| 分类 | Architecture / Engine |

## 一句话结论
主张用单发布者-多订阅者消息队列重构分布式推理 RPC:批量、异步传参,把调度下沉到 worker,支持多模型进程组灵活共存。

## 提案动机
当前分布式推理架构不够灵活:难以添加不同 TP size 的投机解码(PR #5414);RLHF 框架想与 vLLM 进程同步权重也很困难(#5723)。作者分析:与处理静态数据、SPMD 风格的分布式训练不同,推理要处理动态数据(web 请求或用户随时发送的数据),本质是 RPC 问题——需要回答如何启动模型进程组、空闲时做什么、如何感知请求、如何取参数、如何返回输出。

## 提出的方案
- 回顾架构演进:早期 ray 顺序传参,TP 大时 rpc 延迟可超过模型执行本身 → PR #2221 合并 server 与 rank0 TP 进程、用 broadcast 传参(报告超 50% 吞吐提升),但牺牲灵活性 → PR #4894 滥用 broadcast 同时发起 rpc 调用与传参(易超时)→ PR #5399 增加共享内存传输
- 提议 message-queue 方案:单发布者(server)-多订阅者(model processes)队列做输入;节点内用共享内存(publisher 写一次,worker 直接读),跨节点经独立消息服务器 push/pull;可混合两者(同节点走共享内存,利于 PP)
- 需要的 RPC 特性:批量 rpc 调用(发起 n 个与 1 个成本相同)、批量参数传递(发给 n 个进程与 1 个成本相同)、异步调用(结果可稍后取)
- 输出侧用简单队列即可(server 空闲时取回);RPC 协议 {method, args, kwargs} 可扩展到 add_lora 等任意函数
- 主要挑战:重设计发给 worker 的输入——理想只发 token ids,scheduler/block manager 下沉到 worker,server 进程尽量轻量

## 优化点与收益
照抄数字:PR #2221 改用 broadcast 传参后报告 "over 50% throughput improvement"(替换 ray 顺序传参)。

## 现状与落地
已关闭(创建 2 天后)。正文反馈期为 June 23 - June 30(可延长),作者强调应逐步演进、不牺牲性能与可用性;正文未写明结论,落地情况需查关联 PR。

## 关联
- PR #2221:https://github.com/vllm-project/vllm/pull/2221
- PR #4894:https://github.com/vllm-project/vllm/pull/4894
- PR #5399:https://github.com/vllm-project/vllm/pull/5399
- PR #5755(跨节点消息服务器,进行中):https://github.com/vllm-project/vllm/pull/5755
- #5723(RLHF 权重同步):https://github.com/vllm-project/vllm/issues/5723

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
