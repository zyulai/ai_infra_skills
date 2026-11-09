# RFC #15351: [Performance][RFC]: Improving  paged attention kernel's performance

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15351](https://github.com/vllm-project/vllm/issues/15351) |
| 作者 | wenqiny |
| 创建日期 | 2025-03-23 |
| 状态 | ⚪ closed(关闭于 2025-03-26) |
| labels | performance |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
提议新增 `paged_attention_v3` kernel 处理 GQA:按 KV 头组织 grid,让共享同一 KV 头的多个 query 头在一个 thread block 内加载一次 key/value,消除重复内存加载。

## 提案动机
作者在 paged attention benchmark(`python benchmarks/kernels/benchmark_paged_attention.py`)上发现:GQA 推理下 `num-query-heads` 默认 **64**、`num-kv-heads` 默认 **8**,即 **8** 个 query 头共享同一 key/value 头;但现有 `v2` kernel 的 grid 为 `(num_heads, num_seqs, max_num_partitions)`。以 `blockIdx.x` 为 1..8 的不同 thread block(query_head_idx)会把第 0 个头的相同 key/value 数据分别加载进不同 SM 的 L1 cache 或寄存器,最多重复加载 **8** 次,浪费巨大。实测 L1↔L2 cache 间流量 **>800 MB**,而实际需要处理的 key + value 只有 **64 MB**。

## 提出的方案
- 引入新 kernel `paged_attention_v3` 专门处理 GQA,grid 改为 `(num_kv_heads, num_seqs, max_num_partitions)`。
- 在一个 thread block 内处理共享同一 key 头的多个 query 头,从而把 key 装入 shared memory 后复用,消除冗余的 **8** 倍内存加载。
- 优化后 L1↔L2 cache 间流量降为 **136MB**。

## 优化点与收益
测试条件:`benchmark_paged_attention.py`,batch_size=8, seq_len=4096, num_query_heads=64, num_kv_heads=8, head_size=128, block_size=16, dtype=half,本地 RTX3080。v2 基线 kernel 运行时间 443.387 us,v3 为 311.017 us;作者总结该 benchmark 性能提升 **30%**(`443 vs 311`)。L1/L2 流量从 >800 MB 降到 136 MB。

## 现状与落地
已关闭(2025-03-26,创建后仅 3 天)。正文给出完整 benchmark 对比;是否以 PR 形式合入 vLLM 需查关联 PR(正文未附 PR 链接)。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
