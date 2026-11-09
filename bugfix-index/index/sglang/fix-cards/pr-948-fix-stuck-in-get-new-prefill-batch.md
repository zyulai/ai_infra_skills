# PR #948: Fix stuck in `get_new_prefill_batch`

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#948](https://github.com/sgl-project/sglang/pull/948) |
| 作者 | hnyls2002 |
| 合并日期 | 2024-08-06 |
| labels | (none) |
| 热度 | 👍 3 · 💬 1 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论
`max_new_tokens=None` 且 `return_logprob=True` 且命中 radix cache 共享前缀时,`last_node` 与 `prefix_indices` 不对齐,radix cache 锁状态算错,server 永远重复 `get_new_prefill_batch` 且不接收任何请求;改为按截断后的 key 做前缀匹配修复。

## 问题现象
server 无限重复 `get_new_prefill_batch` 却一个请求都放不出来(卡死)。触发条件(正文列出四条,须同时满足):1) `max_new_tokens` 设为 `None`;2) max total tokens 小于 context length,即 `max_req_input_len` 由 `max_total_tokens` 决定;3) `return_logprob` 为 `True`;4) server 中已存在该请求可共享的缓存前缀。

## 根因
正文披露:此时 `max_new_tokens` 被设为 `max_total_tokens - len(input_ids) - 1`;现有实现对 `last_node` 与 `prefix_indices` 的处理不对齐——`match_prefix` 用完整 `input_ids` 匹配后再把 `prefix_indices` 截到 `logprob_start_len`,而 `last_node` 仍对应完整匹配,于是在计算新 prefill batch 的可用空间时,radix cache 的锁(lock_ref)状态算错,永远不再接受新请求。

## 修复方案
改 `python/sglang/srt/managers/tp_worker.py` 的 `get_new_prefill_batch`:当 `req.return_logprob` 为真时,先用截断到 `logprob_start_len` 的 `try_match_ids` 去做 `match_prefix`,而不是匹配后再截 `prefix_indices`,保证 `prefix_indices` 与 `last_node` 始终对齐(diff 注释:「the prefix_indices must always be aligned with last_node」):

```diff
+            try_match_ids = req.input_ids
+            if req.return_logprob:
+                try_match_ids = req.input_ids[: req.logprob_start_len]
+            # NOTE: the prefix_indices must always be aligned with last_node
             prefix_indices, last_node = self.tree_cache.match_prefix(
-                rid=req.rid,
-                key=req.input_ids
+                rid=req.rid, key=try_match_ids
             )
-            if req.return_logprob:
-                prefix_indices = prefix_indices[: req.logprob_start_len]
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围;受影响场景为上述四个条件同时成立(scheduler + radix cache 前缀匹配 + logprob 请求)。
- 回归面:仅 `get_new_prefill_batch` 中前缀匹配的 key 构造。`return_logprob=False` 时 `try_match_ids` 即 `input_ids`,行为完全不变;`return_logprob=True` 时匹配范围缩短到 `logprob_start_len`,匹配结果与 `last_node`、`prefix_indices` 一致,radix cache 锁引用计数恢复正确。匹配前缀变短意味着 logprob 请求可能少复用一部分缓存(匹配长度以 `logprob_start_len` 为界),属该函数语义内的一致性取舍。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
