# RFC #57175: [RFC]: Retain MRV2 DBO CUDA graph replay under DP imbalance through real-token staging

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57175](https://github.com/vllm-project/vllm/issues/57175) |
| 作者 | algebra-MCX |
| 创建日期 | 2026-09-16 |
| 状态 | ⚪ closed(关闭于 2026-09-16) |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议在 CUDA Graph replay 前做 real-token staging,把真实 token 重分布进各 microbatch,DP 不均衡时保留 FULL graph replay、避免不必要的 eager 回退。

## 提案动机
作者在研究 MRV2 DBO 下不均衡 DP decode 负载的 eager fallback(基于 #51700 与原 RFC #50738)。例:per-rank token 数为 [63, 97],128-token FULL CUDA graph 的每个捕获 microbatch 容量 64 token;较小 rank 上第一个 microbatch 含 63 real + 1 padding,第二个只有 padding。现有守卫 `2 * min_tokens < descriptor.num_tokens` 因此拒绝 microbatched graph,使整个 DP 组回退 eager。作者想验证能否通过在每个 rank 已捕获的输入布局内重分布真实 token(不在 DP rank 间搬移请求)来保留 FULL CUDA graph replay;动机是在保住 graph 可见地址、物理容量与 collective 顺序的同时避免不必要的 eager fallback。新增的 staging、metadata 更新与输出恢复会引入开销,须与保留 graph replay 的收益权衡评估。

## 提出的方案
- 在 CUDA Graph replay 之前引入 real-token staging:不把 63 个真实 token 全放第一个 microbatch、第二个留空,而是重分布到两个已捕获区域。正文给出的示例布局:

| Captured region | 现有布局 | 提议布局 |
|---|---|---|
| Rows 0–63 | 63 real + 1 padding | 32 real + 32 padding |
| Rows 64–127 | 64 padding | 31 real + 33 padding |

- 复用现有 graph 容量,保证两个 microbatch 都含真实 token。
- 实现必须保持 token-to-request 映射、positions、attention metadata、KV-cache 索引、MoE dispatch/combine 映射与输出顺序。
- 对不支持或不安全的情形保留现有 eager fallback;范围限定于可安全执行重分布的 eligible decode 负载。

## 优化点与收益
正文以表格给出初步吞吐数据(Higher is better;E/G/A 三种配置的含义未在本帖正文文本中定义,环境/硬件信息正文未给出):

| Seed | Config | Duration (s) | Requests/s | Output tokens/s | Total tokens/s |
|---|---|---|---|---|---|
| 1 | E | 271.24 | 3.7752 | 1932.90 | 2416.13 |
| 1 | G | 261.34 | 3.9183 | 2006.16 | 2507.70 |
| 1 | A | 237.66 | 4.3086 | 2206.01 | 2757.51 |
| 2 | G | 263.42 | 3.8873 | 1990.29 | 2487.86 |
| 2 | A | 237.45 | 4.3125 | 2208.02 | 2760.03 |

正文照抄结论:相对 G,Median E2E latency improves by 11.36% and 12.52%。

## 现状与落地
已关闭(关闭于 2026-09-16,与创建同日,0 评论)。Feedback Period 一栏写"至少一周",作者表示希望在进行 production-ready 实现前获得早期设计反馈,基准结果与 profiling 证据将随可得而补充;但帖子当日即被关闭,正文未说明关闭原因或是否有替代方案,落地情况需查关联讨论/PR。

## 关联
- 前置 issue: https://github.com/vllm-project/vllm/issues/51700
- 原 RFC: https://github.com/vllm-project/vllm/issues/50738

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
