# PR #4855: Fix fa3 cuda graph page_size > 1 precision and page_size=1 speed

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4855](https://github.com/sgl-project/sglang/pull/4855) |
| 作者 | qingquansong |
| 合并日期 | 2025-03-28 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
fa3 cuda graph replay 时 page_table 直接填 token 索引且全表重写:page_size>1 精度错、page_size=1 慢;改为除以 page_size 并只更新所需页。

## 问题现象
FlashAttention(fa3)后端在启用 cuda graph 时有两类问题:page_size > 1 时出现精度问题(precision,来自标题),page_size = 1 时速度受损(speed,来自标题)。正文除标题一行外没有任何症状、复现或影响面的描述。

## 根因
正文未披露根因分析(Motivation 仅重复标题)。

从 diff 可以描述改动所暗示的机制:`init_forward_metadata_replay_cuda_graph` 中,旧的 replay 路径把 `metadata.page_table` 直接赋值为 `self.req_to_token[:, strided_indices][req_pool_indices[:bs]]`——填入的是 req_to_token 里的 token 级索引,没有除以 `page_size`;page_size > 1 时 attention kernel 需要的是页(page)索引,填 token 索引即产生错误的 KV 定位,表现为精度问题。同时旧代码每次 replay 都用高级索引整体重建 page_table,新代码改为先按 `max_seq_len_k` 求出实际需要的页数,只 `copy_` 更新前 `max_seq_pages` 列、其余列填 0,避免了整表重写(page_size=1 的速度收益来源)。(由 diff 推断)

## 修复方案
改 `python/sglang/srt/layers/attention/flashattention_backend.py` 的 `init_forward_metadata_replay_cuda_graph`:

```diff
-        metadata.page_table = self.req_to_token[
-            :, self.decode_cuda_graph_metadata["strided_indices"]
+        max_seq_pages = (metadata.max_seq_len_k + self.page_size - 1) // self.page_size
+        page_indices = self.req_to_token[
+            :, self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages]
         ]
-        metadata.page_table = metadata.page_table[req_pool_indices[:bs]]
+        page_indices = page_indices[req_pool_indices[:bs]] // self.page_size
+        metadata.page_table[:, :max_seq_pages].copy_(page_indices)
+        metadata.page_table[:, max_seq_pages:].fill_(0)
```

即:计算本 batch 最大序列所需页数 `max_seq_pages`;取 req_to_token 的对应切片、按 batch 选行后整除 `page_size` 得到页索引;原地拷入 page_table 前段,尾部清零。

## 影响与回归风险
- 回归面:仅影响 flashattention_backend 的 cuda graph replay 元数据路径;page_size > 1 时 page_table 内容从 token 索引修正为页索引(数值语义变化,依赖旧行为的外部逻辑会受影响),page_size = 1 时整除为无操作但更新方式从整体重建变为原地局部拷贝。capture(capture_cuda_graph)路径与其他后端不受影响。

## 关联
正文未列出
