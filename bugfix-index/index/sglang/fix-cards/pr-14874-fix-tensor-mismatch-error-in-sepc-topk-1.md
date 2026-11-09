# PR #14874: Fix tensor mismatch error in sepc + topk > 1 + page_size > 1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14874](https://github.com/sgl-project/sglang/pull/14874) |
| 作者 | ZeldaHuang |
| 合并日期 | 2025-12-14 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

EAGLE spec 解码在 topk>1 且 page_size>1 时页表张量行数不匹配报错,按实际 num_seqs 截断修复。

## 问题现象

 speculative decoding(sepc,即 EAGLE 相关流程)在 `topk > 1` 且 `page_size > 1` 的组合下出现 tensor mismatch(张量尺寸不匹配)错误。正文未描述具体报错信息与复现步骤,Motivation 一节仅引用 issue #14624。

## 根因

正文未披露根因(Motivation 只写了 #14624,Modifications 为空)。

从 diff 描述改动所暗示的机制(由 diff 推断):`flashattention_backend.py` 的 `update_draft_decode_set_expand_metadata_with_page_size` 中,`expand_page_table` 的行数大于实际序列数(expand 扩出的行数与真实 `num_seqs` 不一致)。原先 `expand_page_table -= last_page_lens_broadcast` 把按 expand 行数广播的 `last_page_lens` 与整个(偏大的)页表相减,随后 `page_table[:, :max_seq_pages_expand].copy_(expand_page_table)` 又按列方向整表拷入 `page_table`,在 topk>1 + page_size>1 时两个张量形状对不上,触发 mismatch;按实际 `num_seqs` 截断后形状一致。

## 修复方案

`python/sglang/srt/layers/attention/flashattention_backend.py` 的 `update_draft_decode_set_expand_metadata_with_page_size` 两处改动:

1. 相减前取 `num_seqs = expand_page_table.shape[0]`,只对 `last_page_lens_broadcast[:num_seqs]` 做广播相减;
2. `copy_` 目标从 `page_table[:, :max_seq_pages_expand]` 收窄为 `page_table[:num_seqs, :max_seq_pages_expand]`。

关键 diff 摘录:

```diff
-    expand_page_table -= last_page_lens_broadcast
+    num_seqs = expand_page_table.shape[0]
+    expand_page_table -= last_page_lens_broadcast[:num_seqs]
...
-    page_table[:, :max_seq_pages_expand].copy_(expand_page_table)
+    page_table[:num_seqs, :max_seq_pages_expand].copy_(expand_page_table)
```

同时在 `test/registered/spec/eagle/test_eagle_infer_a.py` 的正确性测试配置中新增一组 `{cuda_graph_bs: [5], page_size: 4}` 覆盖该场景。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 flashattention backend 的 EAGLE draft decode 元数据更新路径;当 `expand_page_table` 行数恰等于 num_seqs 时行为与原来完全一致,差异只出现在扩出行数大于实际序列数的情形,收窄拷贝目标不会写 `page_table` 的多余行。影响面集中在 spec + topk>1 + page_size>1 的组合。

## 关联

- Issue/PR #14624(正文 Motivation 引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
