# PR #29654: [spec] Fix index_share_for_mtp_iteration being a no-op in EAGLE MTP draft

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-06-29 |
| 分类 | Speculative Decoding | tier: major |
| Label | deepseek, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 4 个文件:`forward_batch_info.py`、`deepseek_nextn.py`、`eagle_info.py`、`eagle_worker_v2.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/29654 |

## 一句话结论

修复 EAGLE MTP draft 中 DSA indexer top-k 复用永不生效的 bug,长上下文(ISL 约 500k)单 draft step 提速 1.88x。

## 动机

#28192 引入的 `index_share_for_mtp_iteration` 设计意图是:在第一个 MTP draft step 运行一次 DSA indexer,后续 draft step 复用其 top-k 结果,避免每步重算。但该机制实际是 silent no-op——复用从未生效,每个 draft step 都在重算 indexer。根因在于对象生命周期:每个 draft step 在一份 *copied* ForwardBatch 上 forward(`load_batch` → `extract_buffer`/`replace`),`deepseek_nextn` 里的写回 `forward_batch.topk_indices = topk_indices` 落在这份 per-step copy 上,而 draft loop 跨 step 携带的原始 ForwardBatch 中 `topk_indices` 始终是 None,于是 `prev_topk_indices` 永远拿不到 step 0 的选择。此 bug 长期未被发现,因为短上下文时(ctx < `index_topk`)indexer 会选中全部 token,复用与重算的结果 bit-identical;只有当 top-k 真正发生剪枝(长上下文)时两条路径才分叉。

## 优化点

修复方式是把 top-k 的携带位置从 ForwardBatch 改到 `spec_info`:在 `EagleDraftInput` 上新增 `mtp_topk_indices` 字段。`spec_info` 与既有 `hidden_states` 一样,跨 per-step forwards 以引用共享,写回因此得以存活——step 0 写入后,steps 1-3 经 `prev_topk_indices` 读到同一 tensor,与 deterministic-top-k 设计一致。同时删除已无用的 `ForwardBatch.topk_indices` 字段,`reuse_mtp_topk_indices` 仅保留为门控布尔。`eagle_worker_v2.draft_forward` 在进入循环前把 `spec_info.mtp_topk_indices` 置 None、循环结束后复位。未设置该 flag 的模型完全不受影响:整个路径由 flag(以及 `topk == 1`)门控。这是 bug fix 而非新机制:恢复的是 #28192 既定设计,消除的是每个 draft step 一次的冗余 indexer 前向。

## 关键改动

```text
python/sglang/srt/speculative/eagle_info.py: EagleDraftInput 新增 mtp_topk_indices 字段(跨 draft step 引用共享)
python/sglang/srt/models/deepseek_nextn.py: prev_topk_indices 读参与写回均改走 spec_info
python/sglang/srt/speculative/eagle_worker_v2.py: draft_forward 进入/退出时初始化与复位 spec_info.mtp_topk_indices
python/sglang/srt/model_executor/forward_batch_info.py: 移除死字段 ForwardBatch.topk_indices
```

```python
prev_topk_indices=(
    forward_batch.spec_info.mtp_topk_indices
    if forward_batch.reuse_mtp_topk_indices
    else None
),
)
if forward_batch.reuse_mtp_topk_indices:
    forward_batch.spec_info.mtp_topk_indices = topk_indices
```

## 性能声明

- 单 draft step profile(长上下文 ISL 约 500k):reuse ON(本 PR)2.215 ms vs OFF(main)4.167 ms,**1.88x faster**(约省 47% 时间)——indexer 现在只在 step 0 运行一次,steps 1-3 复用而非逐步重算。增益随 context 缩放(indexer cost 与序列长度成正比):约 500k 时最大,约 80k agentic regime 较小。
- Draft 质量:OpenHands-style 长上下文 agentic workload(约 89k ISL,13-turn)下 accept length 4.64 vs 4.65(有/无 fix),中性,即行为对 draft 质量保持;spec decode 本身 lossless——只改 draft proposal,不改 target verify。
- 测试条件:GLM-5.2-FP8、4x B300、EAGLE `topk=1`、`num_steps=5`、`num_draft_tokens=6`。PR 未给出端到端吞吐数字。

## 验证与回归风险

验证:在上述 GLM-5.2-FP8 / 4xB300 配置下确认复用真正生效——step 0 计算 indexer,steps 1-3 复用,匹配 deterministic-top-k 设计;accept length 对比证明 draft 质量中性。风险面很小:diff 仅 4 个文件、约 20 行;不设 `index_share_for_mtp_iteration` 的模型走原路径(flag 与 `topk == 1` 门控整条链);移除的 `ForwardBatch.topk_indices` 字段此前因 no-op 而实际无人成功消费,但仍需确认无外部读取方;被复用的 topk_indices tensor 跨 3 个 draft step 存活,依赖 spec_info 的引用共享语义不被后续改动破坏。

## 关联

#28192(前序:引入 `index_share_for_mtp_iteration` 机制,本 PR 修复其失效)。无正文列出的其他同分类 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
