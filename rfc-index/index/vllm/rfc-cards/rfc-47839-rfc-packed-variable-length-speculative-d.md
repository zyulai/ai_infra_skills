# RFC #47839: [RFC]: Packed Variable Length Speculative Decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47839](https://github.com/vllm-project/vllm/issues/47839) |
| 作者 | benchislett |
| 创建日期 | 2026-07-07 |
| 状态 | ⚪ closed(关闭于 2026-08-12) |
| labels | RFC, dflash |
| 热度 | 👍 10 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
提议在投机解码中支持批内每请求异构查询长度,按请求选择性验证草稿 token,降低验证开销。

## 提案动机
投机解码中,每步可对批内各请求提交不同数量的草稿 token 做验证,从而节省验证成本;各类选择/调优「验证哪些 token、验证多少」的技巧都共享同一收益——让批内查询长度按请求变长变短,降低验证开销。当前 vLLM 验证输入批的查询长度是同构的,无法享受这一收益。

## 提出的方案
- 支持每个 decode 请求具有异构查询长度的验证输入批;把「总验证 token 预算」与「预算在各草稿 token 间的分配」两个关注点分离:scheduler 只选一个 host 可见的总 token 预算,worker 完全在设备上把预算分配到各请求,避免 query lengths 与 host 之间来回序列化,从而保持异步调度与 CUDA graph 兼容。
- 改动范围:scheduler、model runner 输入准备、CUDA graph 管理、speculator-scheduler 实现(决定提议哪些 token 的逻辑)。
- 基于 ModelRunnerV2 中已有的 batch-level Dynamic Speculative Decoding(PR #45953,已具备 full-graph 兼容)构建。
- 挑战与对策:1) **Varlen kernel 支持**——不少 attention kernel 尚不支持每请求变长查询(如 FlashInfer TRTLLM-Gen MLA Decode、MiniMax M3 Sparse attention、FlashMLA 不支持;FA Decode 经 `flash_attn_varlen_func` 支持;FlashInfer TRTLLM-Gen Decode 支持但未接入 vLLM),需先按后端 gate,后续再逐步铺开;2) **CUDA Graph 兼容**——对每个 `num_reqs`(按 2 的幂/8 的倍数分桶)记录从 1 到 (1 + max_num_speculative_tokens) 的每组 `num_query_tokens_per_request` 图;解耦 `num_query_tokens_per_request` 与 `max_q_len`,统一传 `max_q_len=max_num_speculative_tokens`,用 `cu_seqlens_q` 定位变长查询起点、余下标 `is_padding`、以 0 查询长度请求补齐 `num_reqs_padded`,使 kernel 输入形状与地址恒定;3) **混合批**——FULL CUDA graph 下混合批支持超范围,PIECEWISE 下短期方案是查询块内强制不 padding(预算内全填草稿 token);4) **数据并行**——需在 worker 间同步当前执行的 graph(即 `num_reqs_padded` × `num_tokens_padded` 对),应可自然融入现有 DP sync;5) **Scheduler-Speculator 感知**——扩展 Speculator 接口让 scheduler「查询」提示(如每请求平均应发多少 query token),参考 DSpark 报告用延迟 2 个 engine 迭代的信息做调度,保持异步调度。

## 优化点与收益
收益为减少投机解码的验证开销:批内按请求变长提交草稿 token,避免统一长度带来的验证浪费;设备侧分配预算避免了与 host 的序列化往返,保持异步调度与 CUDA graph 兼容。正文未给出量化数字。

## 现状与落地
已关闭(2026-08-12),labels 含 dflash;正文为设计与征求意见稿(反馈期未填),作者对 scheduler 是否需要参与等问题仍持开放态度。实际落地情况从正文无法看出,需查关联 PR(PR #45953 为其依托的既有工作)。

## 关联
- [PR #45953](https://github.com/vllm-project/vllm/pull/45953)(batch-level Dynamic Speculative Decoding,本 RFC 的构建基础)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
