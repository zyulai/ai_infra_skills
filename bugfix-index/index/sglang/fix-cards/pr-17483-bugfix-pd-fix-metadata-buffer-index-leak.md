# PR #17483: [BugFix][PD]Fix metadata_buffer_index leak when aborted in PD

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17483](https://github.com/sgl-project/sglang/pull/17483) |
| 作者 | ZhengWG |
| 合并日期 | 2026-02-09 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
PD 模式下 waiting_queue 中请求被 abort 时 metadata_buffer_index 未释放导致泄漏;在 abort_request 补上释放逻辑。

## 问题现象
PD(prefill disaggregation)模式下,通过 `abort_request()` 中止请求时,该请求在 `pop_bootstrapped()` 中分配的 `metadata_buffer_index` 不会被释放,造成内存泄漏。触发条件:waiting_queue 中的请求已分配 `metadata_buffer_index`,但在进入 inflight_queue 之前被 abort——正常流程中 index 是在 inflight_queue 阶段被释放的,abort 路径缺少对应释放。泄漏随被中止请求数累积。

## 根因
正文披露:`abort_request()` 处理 waiting_queue 请求时没有释放 `metadata_buffer_index` 的逻辑;index 的正常释放在 inflight_queue 中进行,提前中止的请求走不到该路径,分配的 index 因此永久泄漏。

## 修复方案
1. `python/sglang/srt/disaggregation/prefill.py` 新增辅助函数 `release_req_to_metadata_buffer(req, allocator)`:当请求带有有效的 `metadata_buffer_index`(非 None 且 >= 0)时调用 `allocator.free()` 释放并置为 -1。
2. `process_disagg_prefill_inflight_queue` 中原有"直接 free + 置 -1"两行改用该辅助函数,行为不变。
3. `python/sglang/srt/managers/scheduler.py` 的 `abort_request()` 增加 PREFILL 分支:

```diff
             # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
             if self.disaggregation_mode == DisaggregationMode.DECODE:
                 release_kv_cache(req, self.tree_cache)
+            # For disaggregation prefill mode, free the metadata buffer index
+            if self.disaggregation_mode == DisaggregationMode.PREFILL:
+                release_req_to_metadata_buffer(
+                    req, self.req_to_metadata_buffer_idx_allocator
+                )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅涉及 PD prefill 模式的两条路径——请求中止(`abort_request`)与 inflight queue 完成清理;DECODE 模式原有的 `release_kv_cache` 逻辑未动。辅助函数带 `hasattr`/None/负值三重检查,对未分配 index 的请求是 no-op;释放后置 -1,避免同一 index 被重复释放;与 inflight_queue 路径统一走同一释放函数,逻辑一致。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
