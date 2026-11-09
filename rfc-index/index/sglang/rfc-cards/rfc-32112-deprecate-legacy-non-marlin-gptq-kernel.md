# RFC #32112: Deprecate legacy (non-Marlin) GPTQ kernel and Dual Chunk Flash Attention backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32112](https://github.com/sgl-project/sglang/issues/32112) |
| 作者 | b8zhong |
| 创建日期 | 2026-07-22 |
| 状态 | ⚪ closed (2026-09-10) |
| labels | RFC |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Quantization |

## 一句话结论
提议废弃非 Marlin 的老式 GPTQ CUDA kernel 与 Dual Chunk Flash Attention 后端,精简量化/注意力路径。

## 提案动机
降低 SGLang 量化与 attention backend 路径的复杂度,移除两块低维护价值的旧实现。

## 提出的方案
- GPTQ (non-Marlin):原始 desc_act/exllama 风格 GPTQ CUDA kernel(sgl-kernel/csrc/gemm/gptq 中的 gptq_gemm、gptq_shuffle),只在 GPTQ checkpoint 不兼容 Marlin 时才会走到。GPTQMarlinConfig 已默认把任何 Marlin 兼容 checkpoint 自动升级到 gptq_marlin(活跃维护、已完全迁移到 JIT);该 plain kernel 只服务 Marlin 支持的 bits/group_size/desc_act 组合之外、不断缩小的 checkpoint 集合,且自身没有 JIT/Triton 替代。
- Dual Chunk Flash Attention:支撑 --attention-backend dual_chunk_flash_attn 的稀疏注意力 kernel(fwd_sparse、varlen_fwd_sparse、convert_vertical_slash_indexes/_mergehead),为 Qwen/Qwen2.5-14B-Instruct-1M 的 1M 上下文支持而加。纯 opt-in:没有任何模型注册或自动选择该后端;底层 kernel 约一年无功能性改动,现在有更多高效稀疏注意力选择。
- 反馈期:3 周。

## 优化点与收益
正文未给出量化数字;收益形式为减少量化与注意力后端路径的复杂度和维护面。

## 现状与落地
已关闭(2026-09-10)。正文未写明废弃是否已执行、由哪个 PR 承接,落地情况需查关联 PR。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
