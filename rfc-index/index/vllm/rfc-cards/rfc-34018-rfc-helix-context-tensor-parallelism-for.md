# RFC #34018: [RFC]: Helix (Context + Tensor) Parallelism for Efficient Long-Context Decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34018](https://github.com/vllm-project/vllm/issues/34018) |
| 作者 | sungsooha |
| 创建日期 | 2026-02-06 |
| 状态 | ⚪ closed(关闭于 2026-03-04) |
| labels | RFC |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |

## 一句话结论
提议把 NVIDIA Helix(Context + Tensor)并行引入 vLLM,消除长上下文解码中 TP > num_kv_heads 时的 KV cache 复制。

## 提案动机
现代 LLM 普遍用 MLA/GQA 压缩 KV cache,KV 头数很少:DeepSeek-V2/R1(MLA,1 个有效 KV 头)在 TP=8 下有 8× KV 复制;Llama-3 70B/405B(GQA,8 头)在 TP=32 下 4× 复制;Qwen2.5(GQA,4-8 头)2-8× 复制。当 `TP > num_kv_heads` 时传统 TP 必须跨 GPU 复制 KV cache,浪费解码期内存带宽。vLLM 现有 DCP 沿序列维分片 KV cache,但在标准 DCP(TP=32、K=8、DCP=4)下每个 rank 有 TPA=8 个 Q 头却只有 HKVP=2 个 KV 头,AllGather Q 后每个 rank 用本地 2 个 KV 头算全部 32 个 Q 头,等于每个 KV 头跨 TPA 复制 4×,长上下文(256K+ tokens)下内存压力大。

## 提出的方案
- Helix 将 attention 与 FFN 的并行方式解耦,同一批 N 张 GPU 每层在两种配置间"扭转":Attention 阶段 KVP × TPA(序列分片 × 头分片),FFN 阶段 TPF × EP(标准张量/专家并行)。
- 与 DCP 的关键区别:通信用 **All-to-All + LSE** 替代 AllGather Q + ReduceScatter;Q 按 TPA 分布而非 AllGather 到所有 rank;KV 零复制(TP=32、K=8 时 TPA=8、KVP=4 即可)。
- 数据流:QKV Proj(冗余)→ FlashAttn(本地 KV)+ LSE → All-to-All 交换 → LSE reduction(精确 softmax)→ Output Proj → AllReduce;All-to-All 交换带 log-sum-exp 的部分注意力输出,可数学精确地跨 KVP rank 重建注意力。
- 设计原则:仅加一个 `helix_mode` 开关,复用 `tensor_parallel_size` 与 `decode_context_parallel_size`;复用 DCP 基础设施(KV 交错、slot mapping、rank groups);与 DCP 支持相同模型/后端。
- 路线图:Phase 1 MLA 核心 → Phase 2 GQA 支持、Phase 3 A2A 优化(可并行)。
- 备选方案被否:扩展 DCP 支持 TP > num_kv_heads 需复制 KV 头,背离 context parallelism 初衷;ring attention 在 decode 阶段因顺序 send/recv 链延迟更高(NVSwitch 全互联下 All-to-All 延迟更低)。

## 优化点与收益
NVIDIA 论文结果(引用自正文,条件为论文所述):固定 batch size 下解码延迟最多降低 **1.5×**;同等延迟目标下并发用户最多 **32×**;通信开销 O(B × H),与序列长度无关。对 vLLM 的收益:消除 TPA > num_kv_heads 时的 KV 复制、缓解长上下文(256K+ tokens)内存压力、让 MLA(DeepSeek)与 GQA(Llama、Qwen)模型扩展更高效。

## 现状与落地
已关闭(2026-03-04)。正文未说明实现进度,落地情况需查关联 PR。

## 关联
- Helix 论文:https://arxiv.org/abs/2507.07120(NVIDIA,2025 年 7 月)
- [#22693 Context Parallelism && Sequence Parallelism](https://github.com/vllm-project/vllm/issues/22693)
- [#26133 Context Parallelism with Fully Sharded KV Cache and Ring Attention](https://github.com/vllm-project/vllm/issues/26133)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
