# RFC #32733: [RFC]: [P/D] Prefill compute optimizations with bi-directional KV cache transfers between P and D nodes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32733](https://github.com/vllm-project/vllm/issues/32733) |
| 作者 | snadampal |
| 创建日期 | 2026-01-20 |
| 状态 | ⚪ closed(关闭于 2026-06-15) |
| labels | RFC |
| 热度 | 👍 2 · 💬 16 |
| 分类 | KV Cache / Memory |

## 一句话结论
提议在 P/D 分离架构中让 P、D 节点间支持双向 KV cache 传输,消除多轮对话与缓存被逐出场景下 P 节点的冗余 prefill 重算。

## 提案动机
当前 vLLM 基于 Nixl KV connector 的 P-D 分离架构中,KV 传输是单向的:P 节点只作生产者、D 节点只作消费者。这在两个关键场景造成低效。其一是多轮对话:上一轮模型生成回复的 KV cache 只存在于 decode 节点上,P 节点无法访问(假设没有外部共享 KV 存储),导致每轮对话 P 节点都要对全部 response token 重算 KV 投影,对生成长推理链的推理密集型模型浪费尤其明显。其二是 P 节点缓存被逐出:D 节点维护会话状态而 P 节点不维护,活跃会话在 P 节点上被逐出概率更高,cache miss 时 P 节点要重算整个对话上下文(系统提示、历史用户提问、历史模型回复)的 KV 投影。

## 提出的方案
- 引入 P、D 节点间的双向 KV cache 传输能力,使双方可互相加载 KV cache,消除不必要的重算。
- 扩展 Nixl KV connector,使其在 prefill 与 decode 两侧都能处理远端 KV block 并相应更新 external/new token block;本地与远端 block 数更新后,其余 KV 加载流程在两类节点上走相同路径,复用现有 KV load 传输函数。
- 原本提出「新增 API 向 D 节点查询 KV 参数」,但在基准测试中发现额外往返查询 KV block 元数据会引入延迟开销,改为在 D 节点的流式响应中直接返回 kv_transfer_params,省去单独查询。
- proxy 作为轻量有状态路由器:按对话历史在内存中缓存 kv_transfer_params,从 D 节点 SSE 流式响应中内联解析获得;首轮 cache miss 时 P 全量计算 KV,后续轮次命中时 proxy 把上一轮 block ID 传给 P,P 经 RDMA 直接从 D 拉取已有 KV,只计算新 token。proxy 不查询 D 的缓存状态、不持有 KV 数据、不给关键路径增加额外跳数。
- KV cache manager 区分节点专属 KV cache(本地 + 专属外部存储)与共享 KV cache(跨节点共享外部存储),避免 P、D 与共享存储之间的两跳传输;该优化可推迟到后续实现阶段。
- 已评估并否决的替代方案「在 decode 节点上处理增量 prefill」:会破坏 P/D 分离的架构初衷,使 D 节点无法利用专用 prefill 算力,且 D 节点要同时承载 prefill 与 decode 两条执行路径,复杂度高。

## 优化点与收益
正文未给出量化数字。收益为消除两类冗余 prefill 计算:多轮对话不再每轮重算全部历史 response token 的 KV 投影;P 节点 cache 逐出后可从 D 节点直接拉取已有 KV,避免重算整个对话上下文。另有一个工程性数字:放弃独立查询 API 的原因是基准测试发现额外往返会引入延迟开销(未给具体数值)。

## 现状与落地
已关闭(2026-06-15)。正文附有实现 PR 链接(#32553),说明已提交实现代码;该 PR 的合入与落地情况需查关联 PR。

## 关联
- 实现 PR: https://github.com/vllm-project/vllm/pull/32553

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
