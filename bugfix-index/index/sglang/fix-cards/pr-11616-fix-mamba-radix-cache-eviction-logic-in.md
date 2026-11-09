# PR #11616: Fix mamba radix cache eviction logic in `alloc_req_slots`

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11616](https://github.com/sgl-project/sglang/pull/11616) |
| 作者 | rogeryoungh |
| 合并日期 | 2025-10-23 |
| labels | high priority, run-ci |
| 热度 | 👍 5 · 💬 1 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
#11313 重构把 `alloc_req_slots` 移到 `common.py` 后丢了 mamba eviction 调用;本 PR 把驱逐逻辑随迁并在调用点传入 `tree_cache` 恢复。

## 问题现象
mamba/hybrid 路径下 `alloc_req_slots` 中的 cache eviction 逻辑未被调用。该驱逐逻辑由 #11214 引入,实现原本位于 `schedule_batch.py`;由于 #11313 的近期重构把 `alloc_req_slots` 及其关联逻辑移到了 `common.py`,驱逐逻辑没有被执行。该问题也在 [#11214 的评论](https://github.com/sgl-project/sglang/pull/11214#issuecomment-3396063269) 中被报告。后果按逻辑推断为 mamba pool 不足时不会驱逐、alloc 可能失败(正文未描述具体报错;推断自 diff)(由 diff 推断)。正文未提供精度或性能测试数据(Accuracy Tests / Benchmarking 章节为空模板)。

## 根因
正文披露:#11214 引入的 mamba cache eviction 步骤实现于 `schedule_batch.py` 的 `alloc_req_slots`;#11313 重构将 `alloc_req_slots` 及其关联逻辑移动到 `common.py` 时,这段驱逐逻辑没有被随之迁移,导致重构后驱逐不再被调用。

## 修复方案
- `python/sglang/srt/managers/schedule_batch.py`:删除本地(已不再被调用路径使用的)`alloc_req_slots` 方法及其驱逐逻辑,并移除不再需要的 `HybridReqToTokenPool` 导入。
- `python/sglang/srt/mem_cache/common.py`:为模块级 `alloc_req_slots` 增加 `tree_cache: BasePrefixCache | None` 参数;当 `req_to_token_pool` 是 `HybridReqToTokenPool` 且 mamba pool 可用量不足 `num_reqs` 时,若 `tree_cache` 是 `MambaRadixCache` 则调用 `tree_cache.evict_mamba(mamba_num)`(mamba_num = max(0, num_reqs - available_size)),然后再 alloc;新增 `MambaRadixCache` 导入。
- `alloc_for_extend` 调用点改为传入 `batch.tree_cache`。

关键 diff 摘录(common.py):
```diff
+        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
+        if mamba_available_size < num_reqs:
+            if tree_cache is not None and isinstance(tree_cache, MambaRadixCache):
+                mamba_num = max(0, num_reqs - mamba_available_size)
+                tree_cache.evict_mamba(mamba_num)
```

## 影响与回归风险
- 影响窗口:#11313 重构合入之后至本 PR 之前,使用 HybridReqToTokenPool / MambaRadixCache(mamba hybrid 模型)的路径;正文未给出具体版本号。
- 回归面:模块级 `alloc_req_slots` 签名新增必填参数 `tree_cache`,所有调用方需同步更新(diff 中已更新 `alloc_for_extend` 一处;其他潜在调用方不在本 diff 范围);驱逐条件与 #11214 原逻辑一致(仅 MambaRadixCache 且可用量不足时),非 hybrid 路径行为不变。

## 关联
- [#11214](https://github.com/sgl-project/sglang/pull/11214)(引入驱逐逻辑的 PR)及其[评论 3396063269](https://github.com/sgl-project/sglang/pull/11214#issuecomment-3396063269)
- [#11313](https://github.com/sgl-project/sglang/pull/11313)(重构导致逻辑丢失的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
