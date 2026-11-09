# RFC #13020: [RFC]: Async KV Cache Transfer for Disaggregated Inference

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#13020](https://github.com/vllm-project/vllm/issues/13020) |
| 作者 | VertexC |
| 创建日期 | 2025-02-10 |
| 状态 | ⚪ closed(关闭于 2025-07-02) |
| labels | RFC, stale |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Disaggregated Serving |

## 一句话结论
AWS Neuron 团队分享 P/D 分解推理下的异步 KV cache 传输 PoC,让 decode 在接收 prefill 侧 KV 的同时继续执行。

## 提案动机
P/D 分解推理中,decode 侧需要接收 prefill worker 产出的 KV cache。来自 AWS Neuron 推理团队的作者分享了异步 KV cache 传输(roadmap #10818 中提及)的初步 PoC:基于 v0.7.0 引入的 KVCacheTransferAgent 与 v0 scheduler 开发,目标是让 decode 在接收 KV 的同时继续执行,而不是被传输阻塞。

## 提出的方案
- KVCacheTransferAgent(decode worker,KVLookupBuffer 层):维护含 {input_ids, roi, keys, caches, hidden} 条目的 receiver_buffer;新增 async_drop_select API——与触发即时阻塞查找的 drop_select 不同,入队 drop_select_request 后立即返回;专用 drop_select_requester 线程处理队列,经 prefill worker 的 drop_select_handler 发起查找与传输,完成后把结果写入 receiver_buffer
- Scheduler(decode worker):新增 transfer queue 与 TRANSFERRING 状态;把原 _schedule_prefills 拆成两段——_schedule_wait(处理 waiting 队列、分配 KV 内存,prompt 可调度即触发 async_drop_select 并移入 transferring 队列)与 _schedule_transferring(经 receiver_buffer 状态确认传输完成后移入 running 队列);没有就绪传输时,调度器优先既有 decode 请求
- model_runner:prompt 的 KV 始终已在本地内存,直接从 receiver_buffer 取"缓存"条目,不再调用 drop_select

## 优化点与收益
正文未给出量化数字:已在 AWS Trn1/Trn2 实例上用 Neuron 后端(neuronx-distributed-inference)完成初步验证,性能数据待后续分享。

## 现状与落地
已关闭(2025-07-02),labels 含 stale;正文未说明是否实现,落地情况需查关联 PR。作者提及愿就 #10818 中 Alibaba Cloud 关于异步传输与角色切换的思路与社区协作。

## 关联
- #10818(roadmap,含 async KV transfer 讨论)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
