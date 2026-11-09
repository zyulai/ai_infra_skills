# RFC #31064: [RFC]: Optional Hidden States Transfer in `KVConnectorBase_V1` to Eliminate Decoder Prefix-Prefill in P/D Disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31064](https://github.com/vllm-project/vllm/issues/31064) |
| 作者 | xinyu-intel |
| 创建日期 | 2025-12-20 |
| 状态 | ⚪ closed(关闭于 2025-12-20) |
| labels | RFC |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Disaggregated Serving |

## 一句话结论
给 `KVConnectorBase_V1` 增加可选的 last-token hidden states 传输,免去 P/D 分离下 decode 侧前缀预填充,稳定解码时延。

## 提案动机
vLLM v1 的 P/D 分离通常只向 decode worker 传输 KV cache,decode worker 仍需重算 last-token hidden states 才能采样首个输出 token——目前靠调度 decoder-side prefix prefill 实现。该重算把 prefill 式计算注入解码时间线:硬件事件轨迹显示稳态 decode step 为几十毫秒,而 prefix-prefill 注入带来数百毫秒到秒级的 step-time 尖峰,直接抬高 TPOT 与尾 ITL,并发越高、上下文越长越严重。此外对 MLA 模型常用的 DP decode,prefix prefill 也按 DP 调度,实际只需一个 DP rank 产出 hidden states,其余 rank 做冗余或空转工作,带来不必要的 token dispatch 开销与调度器压力。动机场景:在 Intel 加速器(Gaudi、Intel XPU)上部署 DeepSeek-R1(MLA attention),Prefill 用 TP、Decode 用 DP、NIXL connector + GDR,观察来自 8k/1.5k serving 任务中 DP16 decode rank 的折线图。

## 提出的方案
- 对 `KVConnectorBase_V1`(仅 v1)做可选、向后兼容的增强:prefill worker 在传输 KV cache 块之外,同时传输待解码 token 的 last-token hidden states。
- decode worker 收到后仅做首 token 采样即进入稳态解码,不再调度 decoder-side prefix prefill;保持现有 async scheduling 语义不变。
- 张量定义:`hidden_states_last`,形状 `(num_tokens, hidden_dim)`,即每个待解码 token 对应的"最后一个 prefill token"的最终层 hidden states,dtype 为模型激活 dtype(bf16/fp16,与 `lm_head` 输入一致);用 `num_tokens` 而非 `batch_size` 以匹配 vLLM 的 token 中心调度(chunked prefill、prefix caching、调度打包都会使其不同于请求数)。
- TP prefill:hidden states 可能在 TP rank 间分片,由指定 producer rank(如 TP rank 0)物化完整 `(num_tokens, hidden_dim)` 张量后传输,避免 decode 侧需要 TP collectives。
- DP decode:单个 DP rank(如 local rank 0)计算 logits 并做首 token 采样,采样出的 token ID 用现有机制广播给其他 DP rank。
- 配置与兼容:opt-in(默认关闭);关闭或不可用时行为不变;运行时 hidden states 传输失败则回退到现有 decoder-side prefix prefill 路径以保正确性。
- 生命周期:hidden states 是临时模型输出、不进现有 cache allocator;按 request/token 作用域管理、采样后确定性回收;可参考 NIXL encoder cache 注册式的预注册 connector 管理临时缓冲,具体机制留待实现。

## 优化点与收益
正文未给出量化数字(正文图中尖峰为定性描述:稳态几十毫秒、注入后数百毫秒到秒级)。收益:消除解码管线中断、降低 TPOT 与尾 ITL;DP decode 下消除 MLA 的冗余/空转计算与 token dispatch 开销,让解码回到 P/D 分离本来的目标——稳态。

## 现状与落地
已关闭(关闭于 2025-12-20,与创建同日)。正文无结论性信息,落地情况需查关联 PR。

## 关联
- [#30112](https://github.com/vllm-project/vllm/issues/30112)(Elastic DP Rank Recovery RFC)
- [vllm-project/vllm-omni#62](https://github.com/vllm-project/vllm-omni/issues/62)(OmniConnector 全模态分离 RFC)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
