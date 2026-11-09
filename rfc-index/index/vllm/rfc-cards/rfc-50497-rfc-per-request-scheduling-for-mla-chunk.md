# RFC #50497: [RFC]: Per-request scheduling for MLA chunked context (prefill)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50497](https://github.com/vllm-project/vllm/issues/50497) |
| 作者 | LucasWilkinson |
| 创建日期 | 2026-07-31 |
| 状态 | ⚪ closed (2026-08-06) |
| labels | RFC, k3, kimi |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Overlap / Scheduler |

## 一句话结论
把 MLA chunked prefill 的 batch-column 均分调度改为按请求的扁平 chunk 列表,修复异构批次下的低利用率与 workspace 膨胀。

## 提案动机
现有调度把 workspace 在所有 prefill 间均分,每次迭代处理每个 prefill 的同一 context 窗口。问题一,异构批次调度差:64k workspace、contexts `[60k, 1k × 7]` 时 `mcc = 8k` 共跑 8 次迭代,第 2–8 次只处理长请求的行,每次 launch 的 workspace 利用率约 12%,且对所有 prefill 跑 attention/merge 并额外付 `mask_empty_context`。问题二,workspace 内存与 `max_num_seqs × block_size` 耦合:Kimi K3 上 HMA 把 attention block size 抬到 1536 @ TP8、约 11000 @ DEP16,DEP16 + `max_num_seqs=32` 时下限要求约 352k 行,是 64k cap 的 5.5×,用户被迫为与调度器无关的原因手调 `max_num_seqs`。

## 提出的方案
- 用 per-request chunk 的扁平列表(`ContextChunk` dataclass,含 per-request 偏移、DCP per-rank 字段等)替换 `[num_chunks × num_prefills]` 网格,对齐 indexer/sparse-MLA 的形状。
- 策略:贪心整请求打包,装不下才拆。Phase 1 仅当单请求超过 workspace 才拆;Phase 2 再拆尾请求填满 workspace(最优 chunk 数 `ceil(Σ context / W)`)。
- 代价是索引簿记:chunk 不再与 suffix partial 对齐,context 写/merge 变成 per-token-slice。

## 优化点与收益
每次 launch 满用 workspace,attention/up-projection/merge 只计入 chunk 内请求;merge 工作量从 `num_chunks × all_prefill_tokens` 降到 `Σ chunk tokens`;`has_empty_context`/`mask_empty_context` 可删除;workspace 下限从 `max_num_seqs × block_size` 降到一个页(Kimi K3 @ DEP16 约 5.5× less),且与 `max_model_len` 无关;大 page 下优雅降级而非断言。

## 现状与落地
已关闭(2026-08-06);正文与结尾看不出是否实现,落地情况需查关联 PR。

## 关联
正文未列出(仅引用仓库内代码 `mla_attention.py`、`determine_chunked_prefill_workspace_size()` 等实现位置)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
