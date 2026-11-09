# PR #12583: [Deepseek V3.2] Fix accuracy bug in the Indexer

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12583](https://github.com/sgl-project/sglang/pull/12583) |
| 作者 | hlu1 |
| 合并日期 | 2025-11-12 |
| labels | deepseek, high priority, run-ci, sgl-kernel |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
DSV3.2 NSA Indexer 构造 ks/ke 共用同一 offset 导致 logits 边界错位,且未把 ks 传给 topk kernel;拆分 q/k 两个 offset 并透传 row_starts。

## 问题现象
DeepSeek V3.2(NSA 后端)精度下降;同时本 PR 还修了 issue #12190 中的 MTP 崩溃。修复后精度:8×B200 上 GSM8K 20 shots 0.953、30 shots 0.928,GPQA 8 次重复 mean 0.790;开 MTP(EAGLE)后 GSM8K 20 shots 0.947、GPQA mean 0.795;8×H200 上 GPQA mean 0.802。

## 根因
正文披露:Indexer 有两个 bug——(1) 没有把 `ks` 传给 topk kernel;(2) 构造 `ks` 和 `ke` 时使用了同一个 `offset`,而这里应该是两个不同的 offset(diff 中的图示说明:logits 矩阵按 q 行排布但每行的 key 区间由 kv 累积位置决定,例:extend_seq_len=[3,2]、seq_lens=[10,4] 时应为 ks=[0,0,0,10,10]、ke=[8,9,10,13,14],旧实现把 q 累积位置当 kv 累积位置用,区间错位)。

## 修复方案
`python/sglang/srt/layers/attention/nsa/nsa_indexer.py` `_get_topk_ragged`:
1. 单一 `offset` 拆为 `q_offset`(按 extend_seq_len 累积)与 `k_offset`(按每个请求的 seq_len 累积);`ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]`,循环尾分别 `q_offset += extend_seq_len`、`k_offset += seq_len`;
2. `fp8_mqa_logits` 的 q/weights 切片改用 `q_offset`,并新增断言 `logits.shape[1] == k_offset`;
3. `metadata.topk_transform(logits, self.index_topk, ks=ks)` 传入 ks。
`python/sglang/srt/layers/attention/nsa_backend.py` `topk_transform`:签名增加 `ks` 参数,非 fused 分支 `fast_topk_v2(..., row_starts=ks)`,PAGED/RAGGED fused 分支同样传 `row_starts=ks`。另更新 DSV3.2 测试(gsm8k num_shots 8→20、接受长度阈值 2.9→2.7、新增 fp8 kv cache 测试类)与 CI 超时(20→30 分钟)。

```diff
-        offset = 0
+        q_offset = 0
+        k_offset = 0
...
-            ks = torch.full((extend_seq_len,), offset, dtype=torch.int32, device="cuda")
-            ke = ks + seq_lens_expanded[offset : offset + extend_seq_len]
+            ks = torch.full((extend_seq_len,), k_offset, dtype=torch.int32, device="cuda")
+            ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
```

## 影响与回归风险
- 影响窗口:正文未说明;影响 DeepSeek V3.2 NSA Indexer 的 ragged topk 路径(即 issue #11629 修复的第二部分)。
- 回归面:nsa_indexer 的 `_get_topk_ragged`、nsa_backend 的 `topk_transform` 三个分支(非 fused / PAGED / RAGGED)及 sgl-kernel 对应 kernel 的 `row_starts` 语义;MTP 路径(issue #12190 的崩溃);相关测试阈值放宽(2.9→2.7)可能掩盖未来轻微回归。

## 关联
- https://github.com/sgl-project/sglang/pull/12582 (Part 1,需先合入)
- https://github.com/sgl-project/sglang/issues/11629 (修复对象)
- https://github.com/sgl-project/sglang/issues/12190 (MTP crash 一并修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
