# PR #15009: Fix regression caused by fa3 block_table

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15009](https://github.com/sgl-project/sglang/pull/15009) |
| 作者 | wenscarl |
| 合并日期 | 2025-12-13 |
| labels | run-ci, nvidia, hopper |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
numpy 数组索引 torch block_table 触发 PyTorch 上游回归,fa3 decode 吞吐大跌;先转 torch 张量再索引修复。

## 问题现象
Llama4 在 DGXH100 上性能回退 14.23%(issue #14665)。作者用 `bench_one_batch`(meta-llama/Llama-4-Scout-17B-16E,`--attention-backend fa3`、batch-size 128、tp-size 8、input/output 1000)复现:修复前 Decode median latency 0.11281 s / 0.15820 s(median throughput 1134.62 / 809.09 token/s),Total throughput 23253.82 / 1598.95 token/s;修复后 Decode median latency 0.01744 s / 0.01828 s(median throughput 7338.32 / 7003.46 token/s),Total throughput 49923.53 / 12798.53 token/s。

## 根因
正文未披露根因(Motivation 仅给出 issue 链接)。

(由 diff 推断,摘自 diff 中 NOTE 注释)pytorch/pytorch#160256 引入回归:用 numpy 数组(batch 与 block 索引)去索引 torch 张量(block_table)时性能下降;fa3 的 local attention 虚拟批构造恰好采用这种 numpy 索引写法,导致 decode 路径吞吐大幅回退。

## 修复方案
`python/sglang/srt/layers/attention/flashattention_backend.py` 的 `make_local_attention_virtual_batches`:先把两个 numpy 索引数组转成 torch 张量,再对 block_table 做高级索引,绕开上游回归:

```diff
-    block_table_local = block_table[batch_indices, block_indices].view(
+    batch_indices_torch = torch.from_numpy(batch_indices)
+    block_indices_torch = torch.from_numpy(block_indices)
+    block_table_local = block_table[batch_indices_torch, block_indices_torch].view(
         virtual_batches, -1
     )
```

## 影响与回归风险
- 影响窗口:受 pytorch/pytorch#160256 影响的版本范围,正文未说明。
- 回归面:fa3 attention backend 的 local attention 虚拟批构造路径(`make_local_attention_virtual_batches`)。改动仅替换索引操作所用的数组容器(numpy → torch 张量),索引值与 `block_table_local` 的结果内容不变,不涉及数值语义;decode 每步都会走到该路径,是本次性能恢复的直接受益面。

## 关联
- 修复的 issue:[#14665 [Bug] Llama4 on DGXH100 has regressed by 14.23%](https://github.com/sgl-project/sglang/issues/14665)
- 引用上游 PR:https://github.com/pytorch/pytorch/pull/160256(见 diff 注释)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
