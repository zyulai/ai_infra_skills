# RFC #57177: [RFC]: Retain MRV2 DBO CUDA Graph replay under DP imbalance through real-token staging

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57177](https://github.com/vllm-project/vllm/issues/57177) |
| 作者 | algebra-MCX |
| 创建日期 | 2026-09-16 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 0 · 💬 2 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议在 CUDA Graph replay 前做 real-token staging,把真实 token 重分布进各 microbatch,使 DP 不均衡时仍保留 FULL graph replay 而不回退 eager。

## 提案动机
作者在研究 MRV2 DBO 下不均衡 DP decode 负载的 eager fallback(基于 #51700 与原 RFC #50738)。例:128-token FULL CUDA Graph 分两个 microbatch、各 64 token 容量;当 DP ranks 的 token 数为 [63, 97] 时,较小 rank 的第二个 microbatch 只含 padding。现有守卫条件 `2 * min_tokens < descriptor.num_tokens` 会拒绝 microbatched graph,使整个 DP 组回退 eager 执行——保住了正确性,却在真实 token 明明可以重分布进已捕获输入布局的情况下牺牲了 CUDA Graph replay。目标是不改变捕获图形状与 microbatch 数量,保留 FULL CUDA Graph replay。

## 提出的方案
- 在 CUDA Graph replay 之前引入 real-token staging:不把 63 个真实 token 全放进第一个 microbatch、让第二个空转,而是把真实 token 重分布到两个已捕获区域,复用现有图容量并保证两个 microbatch 都含真实 token。
- 实现必须保持 token-to-request 映射、positions、attention metadata、KV-cache 索引、MoE dispatch/combine 映射以及输出顺序。
- 对不支持或不安全的情形,保留现有 eager fallback;范围限定于可安全执行重分布的 eligible decode 负载。
- 作者给出初步基准:对比 E(原始实现、eager 执行)、G(原始实现、CUDA Graph 开启,MLA 实际为 FULL_DECODE_ONLY、fallback 仍发生)、A(提议的 DeepSeek 适配,CUDA Graph + real-token staging)三种配置。

## 优化点与收益
正文未给出文本形式的量化数字(基准结果以截图呈现,pack 无法照抄具体倍数)。可比对的事实条件:每次 run 处理 131,072 input tokens 与 524,288 output tokens,1,024 requests 全部完成、零失败;A 配置即提议方案在 CUDA Graph + real-token staging 下的表现见正文附图。

## 现状与落地
🟢 open(2026-09-16 创建),labels 仅 RFC,Feedback Period 一栏为 No response,尚在讨论/等待评审阶段;作者已基于 DeepSeek 完成适配原型并跑了初步基准。

## 关联
- 前置 issue: https://github.com/vllm-project/vllm/issues/51700
- 原 RFC: https://github.com/vllm-project/vllm/issues/50738

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
