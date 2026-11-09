# PR #6697: [Bugfix] Fix slice operation when chunk size mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6697](https://github.com/sgl-project/sglang/pull/6697) |
| 作者 | ShangmingCai |
| 合并日期 | 2025-05-28 |
| labels | high priority |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Mooncake KV 传输在 chunk 大小不匹配时把"切片截断"误写成"单元素索引",`prefill_kv_indices` 被错误取值;补上冒号恢复切片语义。

## 问题现象
正文未描述具体现象与复现步骤,仅称这是一处处理 chunk size mismatch 时的小错误:"Minor fix about a mistake when handling a chunk size mismatch. It should be a slice operation in the first place."。(由 diff 推断)触发路径为 PD 分离下 Mooncake 连接的 `transfer_worker`:当本地分块的目的 KV 索引(`chunked_dst_kv_indice`)与待传的 `kv_chunk.prefill_kv_indices` 长度不一致时,进入截断+告警分支,此时代码对 `prefill_kv_indices` 的操作出错,影响该批 KV 索引的后续传输。

## 根因
正文披露:此处本应是切片(slice)操作,却写成了索引操作——`kv_chunk.prefill_kv_indices[len(chunked_dst_kv_indice)]` 取的是"单个第 n 个元素"而非"前 n 个元素",因此截断逻辑完全失效(单元素索引还会让剩余索引列表从张量变标量)。

## 修复方案
`python/sglang/srt/disaggregation/mooncake/conn.py` 的 `transfer_worker` 中,给截断补上切片冒号,使其保留 `prefill_kv_indices` 的前 `len(chunked_dst_kv_indice)` 个元素:

```diff
                             kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
-                                len(chunked_dst_kv_indice)
+                                : len(chunked_dst_kv_indice)
                             ]
```

随后的长度告警日志保持不变。

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面:仅 `disaggregation/mooncake/conn.py` 中 chunk size mismatch 的异常分支;正常长度匹配的传输路径不经过该分支,不受影响。修复后 mismatch 时 `prefill_kv_indices` 被正确截断为与目的分块等长,而不是取第 n 个单元素(按 diff 如实陈述)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
