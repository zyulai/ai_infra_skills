# PR #27291: [HiCache] Fix SWA L3 cache miss due to a prefetch/hit len mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27291](https://github.com/sgl-project/sglang/pull/27291) |
| 作者 | vladnosiv |
| 合并日期 | 2026-06-18 |
| labels | high priority, run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 24 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
SWA+HiCache L3 下 prefetch 与后续匹配的长度口径不一致,导致应命中的请求 L3 miss;prefetch 改为按 `_compute_max_prefix_len` 截断对齐。

## 问题现象
SWA + HiCache L3 下以下场景失败:1) 对 N 个 token 做 prefill(N % page_size == 0);2) 调 `/flush_cache`;3) 对同样 N 个 token 再做 prefill,此时应看到 >= N - page_size 的 cache hit,实际却不命中。

## 根因
正文披露:全 cache 命中(因为 SWA)时,prefetch 从存储取回 N 个 token 建出的树是 `(N - page_size, null) --> (page_size, page_size)`;但 prefetch 之后,匹配过程用到了 logprobs 的修正(正文引用 `schedule_batch.py` 中相关代码行)。由于这个修正,匹配看不到树的叶节点,于是无法命中缓存——因为所需的 SWA 部分并没有被 prefetch 进来。非 SWA 模型没有这个问题,因为它们可以退而使用叶子的父节点。

## 修复方案
修改 `python/sglang/srt/managers/scheduler.py` 的 `_prefetch_kvcache`:把匹配终点对齐到与后续匹配相同的口径,即用 `req._compute_max_prefix_len(len(req.full_untruncated_fill_ids))` 算出 `match_end`,prefetch 范围从 `full_untruncated_fill_ids[matched_len:]` 改为 `full_untruncated_fill_ids[matched_len:match_end]`:

```python
match_end = req._compute_max_prefix_len(
    len(req.full_untruncated_fill_ids)
)
new_input_tokens = req.full_untruncated_fill_ids[matched_len:match_end]
```

修复后 prefetch 建树变为 `(N - 2 * page_size, null) --> (page_size, page_size)`,可命中 `(N - page_size) / (N - 1)` 个 token。同时新增测试 `test_l3_prefetch_full_prefix_hit_after_flush`(flush 后重新请求,断言 cached_tokens 下限),并为 DSV4 Flash / EAGLE / Mamba / GLM 等子类配置 `l3_prefetch_page_size`、`l3_prefetch_prompt_pages`、`l3_prefetch_max_uncached_tokens`(Mamba 因状态只在 chunk 边界持久化,允许最多一个 chunked_prefill_size 的尾部 token 不缓存)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:`_prefetch_kvcache` 对所有启用 HiCache L3 prefetch 的请求都会按新的 match_end 截断 prefetch 范围,prefetch 的 token 数可能减少;测试侧把 mamba 测试的 chunked-prefill-size 参数化。

## 关联
正文未列出 issue/PR 链接(仅引用了仓库内代码行链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
