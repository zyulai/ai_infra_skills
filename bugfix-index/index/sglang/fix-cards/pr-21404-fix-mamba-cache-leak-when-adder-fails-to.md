# PR #21404: fix mamba cache leak when adder fails to add a matched req.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21404](https://github.com/sgl-project/sglang/pull/21404) |
| 作者 | strgrb |
| 合并日期 | 2026-03-30 |
| labels | run-ci |
| 热度 | 👍 4 · 💬 10 |
| 分类 | Model-specific Fast Path |
| bug 类型 | resource |

## 一句话结论
PrefillAdder 加 req 失败时未释放已匹配分配的 mamba 缓存造成泄漏;adder 失败路径补 free 逻辑修复。

## 问题现象
mamba cache 内存泄漏。Scheduler 在空闲自检时抛出异常(照抄关键行):

```
ValueError: token_to_kv_pool_allocator memory leak detected! full_available_size=118191, full_evictable_size=261376, self.token_to_kv_pool_allocator.size=379567, self.tree_cache.full_protected_size()=0
mamba_available_size=700, mamba_evictable_size=129, self.req_to_token_pool.mamba_pool.size=830, self.tree_cache.mamba_protected_size()=0, leaked_full_pages=None, leaked_mamba_pages={0, 57}
```

触发场景:调度器运行中 `check_memory` 检出 `leaked_mamba_pages={0, 57}`。作者指出与 issue #18300 是同一问题。每次泄漏 1 个 mamba(见根因第 4 点)。

## 根因
正文披露的完整链条:
1. `MambaRadixCache` 调用 `_match_post_processor`,其中会调用 `mamba_pool.alloc`;
2. 在 `Scheduler._get_new_batch_prefill_raw` 中,`PrefillAdder` 尝试把 waiting queue 中的 req 加入,调用 `req.init_next_round_input(self.tree_cache)`,若匹配则触发分配;
3. 若 adder 最终未能加入该 req,第 2 步分配的 mamba 就泄漏了;
4. 由于 `batch_is_full` 被置为 `True`,下一批不会重复走到这条路径,因此每次只泄漏 1 个 mamba。

## 修复方案
正文 Modifications:「Add a free logic when adder failed to add a req to avoid leak」。diff 显示在 `_get_new_batch_prefill_raw` 中,req 未被 adder 加入时回滚已分配的 mamba idx:

```diff
+                # revert matched mamba idx to avoid memory leak, if req is not added
+                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
+                if not added and req.mamba_pool_idx is not None:
+                    self.tree_cache.req_to_token_pool.mamba_pool.free(
+                        req.mamba_pool_idx.unsqueeze(-1)
+                    )
+                    req.mamba_pool_idx = None
```

## 影响与回归风险
- 回归面:`_get_new_batch_prefill_raw` 的 adder 失败分支新增 free 操作;判断条件 `len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]` 决定是否回滚,若 req 实际已被加入而被误判为未加入,可能错误释放其 mamba idx(由 diff 推断);`req.mamba_pool_idx` 被置 None,后续路径需容忍该状态。

## 关联
- [#18300](https://github.com/sgl-project/sglang/issues/18300)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
