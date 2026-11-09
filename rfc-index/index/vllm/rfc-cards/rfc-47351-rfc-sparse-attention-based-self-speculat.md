# RFC #47351: [RFC]: Sparse-attention-based self-speculative decoding (StreamingLLM, then Vegas)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47351](https://github.com/vllm-project/vllm/issues/47351) |
| 作者 | npz7yyk |
| 创建日期 | 2026-07-01 |
| 状态 | ⚪ closed(关闭于 2026-10-31) |
| labels | RFC, stale |
| 热度 | 👍 3 · 💬 3 |
| 分类 | Speculative Decoding |

## 一句话结论
提议新增 sparse_attn 自推测解码:目标模型用稀疏 KV 打草稿、再对全缓存并行验证,免独立草稿模型,无损加速。

## 提案动机
长上下文解码通常受 KV cache 访问瓶颈限制。基于稀疏注意力的自推测解码先用目标模型在 KV cache 子集上打草稿,再对全缓存并行验证,无需单独草稿模型、不加权重、不需训练即可无损加速。vLLM 现有 spec-decode 框架覆盖 ngram、Eagle、Medusa、MTP,均需草稿模型或依赖 n-gram 匹配,无一支持同模型稀疏 KV 草稿,本 RFC 补上这一空缺。作者是 Vegas 论文共同作者,已在 vLLM 中完成可运行实现并希望分两个 PR 上游化。

## 提出的方案
- 新增 speculative method `method="sparse_attn"` 与 `SparseAttnProposer`,复用目标模型在稀疏 KV 模式下打草稿,由 `sparse_attn_algorithm` 选择算法。
- PR 1:proposer 框架 + StreamingLLM(sink + 滑动窗口)作为默认稀疏模式。自包含新模块 `vllm/v1/spec_decode/sparse_attn/`(proposer.py + attn_overrider/),外加三处小接线:`vllm/config/speculative.py`(新 method 与字段)、`gpu_model_runner.py`(构造并驱动 proposer)、`scheduler.py`(预留 lookahead slots 以分配草稿 KV 页);不触碰现有 proposer。可运行分支基于 `releases/v0.24.0`(commit ee0da84ab),可干净 forward-port 到 main。
- PR 2:Vegas 算法(verification-guided KV selection),新增 `vegas.py` 及 `varlen_reduce.py`、`varlen_topk.py` 两个 CUDA kernel;另需向 flash-attention 仓库提配套 PR,在 `flash_attn_varlen_func` 上以 `scores` 输出暴露 per-token attention logits。
- 新配置字段:`sparse_attn_algorithm`(默认 `"streamingllm"`)、`sparse_attn_ratio`(打草稿保留的 KV 比例,默认 `0.05`)、`sparse_attn_min_tokens`(每请求 KV 预算下限,默认 `256`)。
- 作者征询的设计点:为保持 proposer 自包含,稀疏 KV 模式目前通过在 draft pass 运行时包裹 attention backend(暂为 FlashAttention)实现,询问维护者是否接受,或倾向在 attention 层提供更正式的 override 机制。

## 优化点与收益
正文给出量化数字:单张 H100、Qwen3-8B、将 AIME 2025 prompts replay 32 次,按累计解码吞吐计,vanilla vLLM 约 ~1,000 tokens/s;StreamingLLM 草稿器(`num_speculative_tokens=4`)约 ~1,600 tokens/s;Vegas(`num_speculative_tokens=6`)约 ~1,850 tokens/s。更多评估结果见 Vegas 论文。

## 现状与落地
已关闭(关闭于 2026-10-31),labels 含 stale。正文写明 PR 1 的可运行分支已就绪、"approach 在此确认后即可提交 PR",PR 2 需配套 flash-attention 改动;正文未写明两者最终是否已提交或合并,落地情况需查关联 PR。

## 关联
- MagicDec: https://arxiv.org/abs/2408.11049
- Vegas 论文: https://arxiv.org/abs/2602.07223
- StreamingLLM(sliding window + attention sink): https://arxiv.org/abs/2309.17453
- 可运行实现 + benchmark: https://github.com/platformxlab/vegas
- PR 1 分支: https://github.com/npz7yyk/vegas/tree/vllm-pr1

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
