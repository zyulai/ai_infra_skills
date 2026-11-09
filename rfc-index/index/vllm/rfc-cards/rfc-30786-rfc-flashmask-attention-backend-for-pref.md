# RFC #30786: [RFC]: FlashMask Attention Backend for PrefixLM Models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30786](https://github.com/vllm-project/vllm/issues/30786) |
| 作者 | lucianommartins |
| 创建日期 | 2025-12-16 |
| 状态 | ⚪ closed(关闭于 2026-04-19) |
| labels | RFC, stale |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Attention / Kernel |

## 一句话结论
提议新增 FlashMask attention 后端,用 O(N) 稀疏掩码加速 PrefixLM/多模态混合注意力,替代慢 2-4x 的 TritonAttention 路径。

## 提案动机
PrefixLM 模型(如 Gemma3 多模态)需要混合注意力:图像 token 双向注意力 + 文本 token 因果注意力。FlashAttention-2 只支持纯因果或纯非因果,无法表达这种混合掩码,迫使 PrefixLM 走更慢的 TritonAttention(FlexAttention 兜底)。在 A100-80GB、seq=512 的基准中:TritonAttention 对 gemma3-4b 为 0.18 ms、gemma3-27b 为 0.16 ms,而 FlashAttention-2 纯文本基线分别为 0.072 ms / 0.082 ms,即 2.5x / 2x 的开销;正文称 2-4x 的开销作用于每一层 attention,累积成显著端到端延迟。

## 提出的方案
- 采用 FlashMask(ICLR 2025 Poster,Wang et al.,"FlashMask: Efficient and Rich Mask Extension of FlashAttention"):以按行的列边界 `col_left`/`col_right` 稀疏表示注意力掩码,内存 O(seq_len) 而非 O(seq_len²)。
- 内核层面在 FlashAttention-2 的 tiling 算法上,把因果判断替换为列边界判断,每元素仍是 O(1)。
- Phase 1:在 `vllm/attention/ops/flashmask/` 实现 CUDA 内核(fwd/bwd/utils),适配 vLLM 张量布局、paged attention block table 与 CUDA graph capture。
- Phase 2:注册 `FlashMaskBackend`(SM 8.0+ 起用),扩展 attention metadata 增加 col_left/col_right 与 per-layer sliding window;在 `gpu_model_runner` 中由 mm_prefix_range 构建列边界;后端选择优先级为 FlashMask > TritonAttention > FlexAttention。
- Phase 3:正确性测试(纯因果对齐 FlashAttention、PrefixLM 对齐 TritonAttention、Gemma3 端到端、滑窗、paged attention)与性能基准;SM < 8.0(如 V100)回退 TritonAttention。

## 优化点与收益
- A100-80GB、seq=512 基准:A图像前缀路径现状 TritonAttention 0.18 ms(gemma3-4b)/ 0.16 ms(gemma3-27b),FlexAttention 0.29 ms / 0.33 ms,相对 FlashAttention-2 基线(0.072 ms / 0.082 ms)为 2.5x / 2x 开销;正文并称该 2-4x 开销逐层累积。
- 目标(测试条件由 RFC 列出):纯文本延迟 ≤ FlashAttention-2(不回退)、多模态延迟 < TritonAttention(提升幅度实现期间确定)、掩码内存 O(N)(以 `torch.cuda.memory_allocated()` 验证)。
- 战略价值:统一文本/多模态单一后端、掩码内存 O(N) vs O(N²) 利于长上下文、可扩展到其他复杂掩码模式。

## 现状与落地
已关闭(2026-04-19),labels 含 stale。正文未说明实现进度,落地情况需查关联 PR。

## 关联
- PR #30386(已为 TritonAttention 增加 PrefixLM 支持)
- FlashMask 论文 "FlashMask: Efficient and Rich Mask Extension of FlashAttention"(ICLR 2025 Poster)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
