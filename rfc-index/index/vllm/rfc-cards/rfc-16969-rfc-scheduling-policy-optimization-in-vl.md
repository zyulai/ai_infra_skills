# RFC #16969: [RFC]: scheduling policy optimization  in vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#16969](https://github.com/vllm-project/vllm/issues/16969) |
| 作者 | RuixiangMa |
| 创建日期 | 2025-04-22 |
| 状态 | ⚪ closed(关闭于 2026-02-23) |
| labels | RFC, stale |
| 热度 | 👍 8 · 💬 9 |
| 分类 | Overlap / Scheduler |

## 一句话结论
提议基于 SJF 思想的调度策略:每轮尽可能多选可运行请求填满 GPU 显存,缓解 FCFS 调度的队头阻塞。

## 提案动机
受论文《Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management》启发:先来先服务(FCFS)调度策略常导致队头阻塞(head-of-line blocking)问题,造成 GPU 显存资源利用不足,在多轮对话使 prompt 长度不断增加时尤为明显。作者据此提出基于 SJF(shortest job first)的调度策略。

## 提出的方案
- 总体设计:在每个调度时刻,从请求队列中选出尽可能多的可运行请求(当前 GPU 显存能容纳的那些),而不是只选单个「下一个可运行请求」,尽量填满 GPU 空闲显存,从而增大 batch size、提升吞吐
- 实现细节:请求排序——队列按所需显存从小到大排序,优先占用显存少的小请求以更好利用碎片显存;贪心选择——从队首依次选取,直到累计所需显存接近/达到 GPU 可用显存上限;动态调整——调度过程中持续更新 GPU 显存状态,按剩余显存重新评估队列中哪些请求可加入当前 batch
- 防饥饿:等待超过阈值的请求提升优先级;某些情况下强制把长请求纳入 batch,即使意味着抢占部分分配给短请求的显存

## 优化点与收益
正文未给出量化数字(定性收益:提高显存利用率、batch size 与吞吐)。

## 现状与落地
已关闭(2026-02-23,label 含 stale)。正文未记录实现与评审结果,落地情况需查关联 PR。

## 关联
正文未列出(动机引用论文但未附链接)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
