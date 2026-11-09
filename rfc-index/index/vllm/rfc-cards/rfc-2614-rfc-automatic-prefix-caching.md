# RFC #2614: [RFC] Automatic Prefix Caching

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#2614](https://github.com/vllm-project/vllm/issues/2614) |
| 作者 | zhuohan123 |
| 创建日期 | 2024-01-26 |
| 状态 | ⚪ closed (关闭于 2024-03-02) |
| labels | RFC, feature request |
| 热度 | 👍 59 · 💬 17 |
| 分类 | KV Cache / Memory |

## 一句话结论
用哈希表而非 Trie 实现 vLLM 自动前缀缓存,让 KV 块像操作系统页缓存一样被共享与淘汰。

## 提案动机
vLLM 需要实现自动前缀缓存:相同前缀的请求应能复用已计算的 KV cache,避免重复计算,提升多请求共享前缀场景下的效率。本文讨论 vLLM 实现该功能的具体计划。

## 提出的方案
- 每个 KV cache 块由 `hash(prefix tokens, tokens in this block)` 唯一标识,在 KV cache 管理中加一层间接:`Logical block table --> hash table --> physical block table`
- 不立即释放引用计数为 1 的块即可实现前缀缓存;每个块维护:hash、引用计数、最后访问时间、总访问次数、块的前缀长度
- 淘汰策略:先只驱逐 ref count == 0 的块,再按最后访问时间 LRU,时间相同则先释放前缀更长的块——与 RadixAttention 策略等价
- 只缓存完整块,不完整块不进哈希表;新增完整块表与不完整块表,不完整块补全时与已有完整块合并或晋升
- 相比 KV block Trie 的优势:可支持非前缀缓存(如 Mistral 滑动窗口只缓存窗口末尾 token、attention sinks 缓存首尾 token)、哈希表维护比树简单、淘汰策略可扩展
- 交付物分 P0(缓存化+淘汰策略+哈希重构)、P1(滑动窗口支持、更快的哈希函数)、P2(针对间隔序列的 kernel 优化、OOM preemption 策略、块交换支持)

## 优化点与收益
正文未给出量化数字。定性收益:跨请求共享前缀 KV、淘汰策略与 RadixAttention 等价且实现更简单、可扩展到滑动窗口/attention sinks 等非纯前缀缓存场景。

## 现状与落地
已关闭。正文为设计计划与 P0/P1/P2 交付清单,未写明实现结论,落地情况需查关联 PR。

## 关联
- RadixAttention 论文:https://arxiv.org/pdf/2312.07104.pdf
- attention sinks 论文:https://arxiv.org/abs/2309.17453

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
