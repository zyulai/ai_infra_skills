# PR #42959: [BugFix][kv_offload]: Prevent offloading stale sliding window blocks

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42959](https://github.com/vllm-project/vllm/pull/42959) |
| 作者 | orozery |
| 合并日期 | 2026-06-02 |
| labels | bug, kv-connector, ready, v1 |
| 热度 | 👍 5 · 💬 2 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
滑动窗口块延迟 offload 期间被释放复用,陈旧 block_id 未被检测;修复为本步重分配命中的陈旧条目置零跳过并 flush 相关 job。

## 问题现象
正文描述:sliding window offloading 的边界场景——滑动窗口块未能立即 offload(例如 CPU 分配失败)时,该块会被 `remove_skipped_blocks` 释放、再重新分配给其他逻辑块;此前这种情况未被检测,可能导致错误地 offload KV 数据,或断言错误(Issue #42761)。正文粘贴的失败日志显示:新增单测在 main 上以与 #42761 相同的错误失败,`_remove_pending_job` 抛 `KeyError: 1`,当时 `block_ids = [1, 2, 3, 1]`(block id 1 重复出现)。

## 根因
正文披露:sliding window 块的 offload 被延迟(如 CPU 分配失败)期间,块被释放并被重新分配,pending store 中记录的 block_id 随之陈旧;重试 offload 时这些陈旧 id 未被检测,会把错误数据 offload 到错误的 key 下,或因同一 block id 在 `_block_id_to_pending_jobs` 中重复而触发 KeyError。

## 修复方案
改 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:
- 新增 `_current_batch_allocated_block_ids` 集合,在 `update_state_after_alloc` 与新抽出的 `_update_req_states`(自 `_build_store_jobs` 拆出)中记录本 engine step 分配的 GPU block id;
- `_update_req_states` 把 sliding window 组中位于 `[next_stored_block_idx * block_size_factor, 本步扩展前长度)` 区间、命中本步新分配集合的 pending-store 位置,陈旧 block_id 置 0;
- `_build_store_jobs` 对 `block_id == 0` 一律跳过(滑动窗口/SSM 跳过或陈旧条目两种含义),并删除「skipped 块不能出现在非 skipped 块之后」的断言;
- 原 alloc/build 阶段对新块与 dst 块 fence pending store 的逻辑移到 `build_connector_meta`,统一 flush 含重分配块的 job;`reset_cache` 增加对应断言;
- 新增回归测试 `test_stale_sliding_window_block_after_prepare_store_failure`(async_scheduling 两种取值)。

关键 diff:
```diff
+                    for j in range(start, end):
+                        if (
+                            group_state.block_ids[j]
+                            in self._current_batch_allocated_block_ids
+                        ):
+                            group_state.block_ids[j] = 0
```

## 影响与回归风险
- 回归面:按 diff 实际改动,集中于 OffloadingConnectorScheduler 的 `update_state_after_alloc`、`_update_req_states`/`_build_store_jobs`、`build_connector_meta`、`reset_cache`。置 0 只作用于 sliding window 组的 pending-store 区间;fence 时点从 alloc 阶段移到 `build_connector_meta`;BLOCK_LEVEL policy 的 `next_stored_block_idx` 逻辑保持不变。

## 关联
- 正文提到 Issue #42761([#42761](https://github.com/vllm-project/vllm/issues/42761)),作者期望本 PR 解决它。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
