# PR #22458: Fix NCCL AllGather hanging issue for Qwen3 Next MTP

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22458](https://github.com/sgl-project/sglang/pull/22458) |
| 作者 | ispobock |
| 合并日期 | 2026-04-10 |
| labels | run-ci |
| 热度 | 👍 4 · 💬 11 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论

EAGLE 投机采样(V1/V2)在 TP>1 + 非 greedy 下,浮点非确定性使各 GPU 采出不同 accepted token,radix cache 前缀分歧导致 AllGather 尺寸不一致而挂起;sampling 后从 rank 0 broadcast 修复。

## 问题现象

EAGLE speculative decoding(V1 与 V2 均受影响)在 TP>1 且使用非 greedy 采样时出现挂起(hanging),直接表现为 LogitsProcessor 中的 AllGather size mismatch,对应 issue #22276;标题指向 Qwen3 Next MTP 场景。

## 根因

正文披露了完整因果链:

1. 非 greedy 采样分支中,各 GPU 拿到相同的 `next_token_logits`,但 `softmax` / `top_k_renorm_prob` / `top_p_renorm_prob` 因浮点非确定性,在不同 GPU 上产生**略有不同**的 `target_probs`;
2. `tree_speculative_sampling_target_only` 因此采出不同的 accepted token values;
3. 分歧的 `output_ids` 被插入 radix cache,使后续请求命中不同的 prefix match 长度;
4. 不同的前缀长度导致不同的 `extend_seq_lens` 与 `num_seqs`;
5. LogitsProcessor 中各 rank 的 AllGather 尺寸不再一致 → hang。

## 修复方案

在 sampling kernel 之后,将采样结果从 rank 0 broadcast 到所有 TP rank,保证各 rank 持有完全相同的 accepted token values,掐断分歧源头。

关键摘录(`eagle_info.py` verify() 与 `eagle_info_v2.py` sample() 各一处,改动相同):

```diff
+            # Sync sampling results across TP ranks: different GPUs may
+            # produce slightly different target_probs due to floating-point
+            # non-determinism in softmax/top_k/top_p, causing different
+            # sampled tokens. Broadcast from rank 0 to ensure consistency.
+            tp_group = (
+                get_attention_tp_group()
+                if is_dp_attention_enabled()
+                else get_tp_group()
+            )
+            if tp_group.world_size > 1:
+                tp_group.broadcast(predict, src=0)
+                tp_group.broadcast(accept_index, src=0)
+                tp_group.broadcast(accept_length, src=0)
```

改动文件仅 `speculative/eagle_info.py` 与 `speculative/eagle_info_v2.py`;DP attention 启用时用 attention TP group,否则用普通 TP group;仅在 world_size > 1 时广播。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,仅 EAGLE V1/V2 的 verify/sample 非贪婪路径;每次采样额外增加 3 次 TP broadcast(predict、accept_index、accept_length),在 TP>1 时引入少量同步开销;单机单卡(world_size=1)与 greedy 路径不受影响。广播后所有 rank 的 accepted 结果一致,radix cache 内容重新收敛为跨 rank 一致。

## 关联

- issue [#22276](https://github.com/sgl-project/sglang/issues/22276)(挂起问题报告)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
