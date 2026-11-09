# RFC #22246: [RFC]: Dynamic Expert Load Balance with Zero-like-overhead

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22246](https://github.com/vllm-project/vllm/issues/22246) |
| 作者 | raindaywhu |
| 创建日期 | 2025-08-05 |
| 状态 | ⚪ closed(关闭于 2026-03-08) |
| labels | RFC, stale |
| 热度 | 👍 1 · 💬 11 |
| 分类 | MoE |

## 一句话结论
提出 SwiftBalancer:异步、近零开销的动态专家负载均衡(EPLB),不再 stop-the-world。

## 提案动机
当前动态专家负载均衡会 stop-the-world。异步的专家负载均衡可避免两类问题:1) Host 侧延迟——EPLB 期间有大量 CPU 操作(eplb 算法、创建 p2p ops、log2phy expert 转换),耗时可达约 1s;2) 通信延迟——无 NVLink 场景下传输耗时显著,且一个专家权重可能要迁移到多个新位置,导致一个专家 N 次 send/recv。作者实测在 Ascend A2 服务器上,传输 16 个专家权重时 `batch_isend_irecv` 耗时超过 100ms。团队已在 vllm-ascend 1943 上完成 SwiftBalancer(vllm-ascend PR #2186)并充分测试。

## 提出的方案
SwiftBalancer 的整体工作流:
1. 前向期间记录专家分布:用 dispatch 之后的 expert_token_num 代替 topk_ids,张量形状更小,降低 HBM 记录与 add 算子的开销。
2. 用 all-gather(而非 all-reduce)同步专家分布,流量更小。
3. 到达 num_iterations 时唤醒 eplb worker 进程,在 eplb worker 中运行 eplb 算法。
4. 生成 p2p send/recv ops 及 log2phy 等其他算子(这些是 CPU 长耗时环节)。
5. 在 forward 之前于 async_stream 中启动 ibatch_send_recv。
6. forward 结束后等待 ibatch_send_recv 完成,再更新 expert map 与专家权重。
配套改进:更低开销的专家分布记录、eplb 算法等 Python 算子的异步 CPU 计算、新 eplb 算法(在效果几乎相同的前提下减少专家重平衡)。

## 优化点与收益
正文给出量化数字(NPU,vllm-ascend 1943,ep_size = 64):每层开销 **1~2ms**,同时带来 **5ms-8ms** 的 decode 延迟收益;对比现状——EPLB 的 CPU 操作可达约 1s,Ascend A2 上 16 个专家权重传输经 `batch_isend_irecv` 耗时超 100ms。整体实现近零开销(Zero-like-overhead)且不再 stop-the-world。

## 现状与落地
已关闭(2026-03-08)。实现已在 vllm-ascend PR #2186 完成并充分测试;正文「Any Other Things」写明当时正在为与 eplb mainline 代码协作添加适配层(adaptor)。是否已进 vLLM 主线从正文无法看出,落地情况需查关联 PR。

## 关联
- [vllm-ascend PR #2186](https://github.com/vllm-project/vllm-ascend/pull/2186)(SwiftBalancer 在 vllm-ascend 1943 上的实现)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
