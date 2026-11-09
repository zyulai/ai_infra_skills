# PR #5173: [Bugfix] Fix index out of bounds in local attention with large sequences

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5173](https://github.com/sgl-project/sglang/pull/5173) |
| 作者 | CatherineSue |
| 合并日期 | 2025-04-09 |
| labels | high priority |
| 热度 | 👍 6 · 💬 7 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
序列略超 attention chunk(如 8193 vs 8192)时 local attention 索引越界;对 block_indices 做 np.clip 修复。

## 问题现象
local attention 场景下,序列长度略大于 attention chunk size 时出现 index error——例如 8193 tokens 配 8192 chunk size,第二个 local attention block 会试图访问超出 block_table 边界的索引。触发条件为长序列(>8192 tokens)或 block_table 维度小于完整 attention chunk 所需尺寸。正文未附具体报错日志。该 PR 修复 issue #5170。

## 根因
正文披露:`make_local_attention_virtual_batches` 生成的 `block_indices` 在展平后可能超出 `block_table` 的维度;当序列略大于 chunk size 时,第二个 local attention block 的索引越过 block_table 边界,造成 index out of bounds。

## 修复方案
`python/sglang/srt/layers/attention/flashattention_backend.py` 的 `make_local_attention_virtual_batches` 中,对展平后的 block 索引增加安全裁剪,保证索引永不超出 block_table 维度:

```diff
-    block_indices = block_indices.flatten()
+    # Ensure block_indices doesn't exceed block_table dimensions
+    # This is a critical safety check that prevents index out of bounds errors
+    # when dealing with large sequences (>8192 tokens) or when the block_table
+    # dimensions are smaller than what would be needed for the full attention chunk size.
+    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:flashattention backend 的 local attention 虚拟批构造路径(`make_local_attention_virtual_batches`)。`clip(max=block_table.shape[1]-1)` 只会改动原本就超出维度的索引值(原先导致越界访问),正常范围内的索引完全不变;chunk 边界附近的长序列(如 8193 tokens)是直接受益场景,其余 batch/序列的行为不受影响。

## 关联
- 修复的 issue:sgl-project/sglang#5170

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
