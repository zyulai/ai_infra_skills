# RFC #50391: [RFC]: Support Speculative Decoding with Decode Context Parallelism

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50391](https://github.com/vllm-project/vllm/issues/50391) |
| 作者 | pisceskkk |
| 创建日期 | 2026-07-30 |
| 状态 | 🟢 open |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Speculative Decoding |

## 一句话结论
让投机解码与 Decode Context Parallelism(DCP)兼容:proposer 按 DCP 拓扑重算 rank-local 长度,attention 后端为每个 query 提供 rank-local KV 边界。

## 提案动机
vLLM 目前不支持投机解码与 DCP 同用(所有 DCP 配置,含 `cp_kv_cache_interleave_size = 1` 与更大的 interleave)。DCP 按全局位置把 KV-cache token 分片到各 rank;投机验证时一个请求含多个 query token,query token `j`(历史上下文长 `H`)必须看到 `local_len(r, H + j + 1)` 个 rank-local KV token——这些边界因 query 与 rank 而异,一般无法从最终 rank-local KV 长度推出。DCP 同样影响 proposer:接受/拒绝后全局序列长度变化,rank-local 长度不能直接加减接受/拒绝数,必须由修正后的全局长度与 DCP 拓扑重新推导。常规 causal attention kernel 从最终 KV 长度推导所有 query 边界,在 DCP 下会暴露未来的投机 KV 或隐藏有效 KV,产生错误的 verifier logits 与接受决策。正文给出 DCP=2、MTP=3、context length=5 的对照示例:正确 per-query 可见性为 rank0 `[3, 4, 4, 5]`、rank1 `[3, 3, 4, 4]`,而 bottom-right causal mask 得到 `[2, 3, 4, 5]` 与 `[1, 2, 3, 4]`。目标是为自回归 proposer、多 token 目标验证、不同上下文长度/DCP rank/interleave/接受结果以及 eager 执行与 CUDA Graph replay 提供完整兼容。

## 提出的方案
1. **Proposer 侧**:每个自回归 draft step 生成 DCP-aware attention metadata;投机拒绝应用后,由全局序列长度、DCP size、DCP rank、interleave size 重算对应 rank-local 长度,禁止 `local_seq_len += 1` / `local_seq_len -= num_rejected` 式更新;每个 draft decode step 前用 `prepare_dcp_local_seq_lens(...)` 填充持久 DCP local-length buffer 并传入 `build_attn_metadata(..., dcp_local_seq_lens=...)`。元数据流程:应用拒绝到全局长度 → 准备绝对草稿位置 → DCP-aware slot mapping → 重算 rank-local 长度 → 构建 draft attention metadata → proposer attention → 为下一步推进全局状态。全局长度每次变化都必须重算。
2. **Attention 后端四策略**(实现 per-query rank-local 可见性):
   - **A 直接 CP-aware attention**:kernel 接收 DCP size/rank/interleave/全局 query 位置与序列长度,一次主 attention call、无自定义 mask、无虚拟 batch 扩展、dispatch 与元数据开销最低——预期性能最好,但内核实现成本最高(FlashAttention 需接口支持 CP rank、CP world size、interleave),每种 interleave 配置都需验证。
   - **B 上下文/当前 KV 拆分**:历史 KV 非因果 + 当前投机块因果,两次 attention call 用输出与 LSE 做 state merge;可按 chunked-prefill 式流程实现,支持任意 DCP rank 与 interleave 布局、无需 rank 特定 dense mask,代价是额外 workspace 与 launch 开销,MLA 实现可能需要 context chunk gathering/重建。可移植性方案。
   - **C 自定义 mask**:按 `local_boundary[j] = local_len(r, H + j + 1)` 构造 rank 特定 mask;保留单次主 attention call、支持任意 DCP size/interleave/可变 query 长度;但需 `O(batch × N × local_KV)` mask 内存、mask 生成成本随上下文增长、增大 CUDA Graph 输入与常驻缓冲,并可能禁用优化的 causal kernel、迫使后端退到通用 masked-attention kernel。适合 custom-mask 支持高效的后端。
   - **D 展开为 N 个虚拟请求**:虚拟行 `j` 的 query position = `H + j`、global seq len = `H + j + 1`、local seq len = `local_len(r, H + j + 1)`、q_len = 1,复用原请求 block table;复用单 token DCP decode kernel、无自定义 mask,但逻辑 batch 至多扩大 `N` 倍、复制 block-table 与序列元数据、降低 batching 效率、需更大 CUDA Graph batch 维度。正确性兜底方案,`N` 越大开销越大。
   - 正文策略对比表(照抄):Direct CP-aware —— 1 次 call、额外元数据 Low、内核改动 High、Expected performance: Best;Context/current split —— 2 + merge、Low、Medium、Good and portable;Custom mask —— 1、`O(B×N×local_KV)`、Low/Medium、Context and kernel dependent;Virtual single-query rows —— 1 expanded batch、Block-table duplication、Low/Medium、Lower for larger `N`。

## 优化点与收益
正文未给出量化数字。收益首先是正确性:修复 DCP 下投机验证的错误 logits/接受决策;性能为各策略间的定性权衡(见策略对比表:Direct CP-aware 为 Best、Context/current split 为 Good and portable、Custom mask 依赖上下文与内核、虚拟行方案在 `N` 大时较低),无实测数据。

## 现状与落地
仍在讨论/等待评审(open);正文的 Feedback Period 为 _No response_,CC 了 LucasWilkinson、GirasoleY、ZJY0516。关联工作显示已有部分先例:#45426 已为 FlashMLA sparse 路径实现 per-token rank-local 边界(interleave=1 时启用 DCP 变长 decode),本 RFC 定位为不假设特定 interleave 的 backend-neutral 设计,尚待按策略推进实现。

## 关联
- #39632(MTP draft steps 间 DCP-aware attention metadata 复用,与 proposer 改动相关)
- #45425(无 DCP 变长 decode 支持的后端在 CUDA graphs 下 MTP 验证结果错误的报告)
- #45426(FlashMLA sparse 路径 per-token rank-local causal 边界实现)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
