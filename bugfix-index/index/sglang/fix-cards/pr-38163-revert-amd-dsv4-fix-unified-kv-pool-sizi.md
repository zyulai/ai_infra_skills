# PR #38163: Revert "[AMD][DSV4] Fix unified-KV pool sizing and SWA ring accounting"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38163](https://github.com/sgl-project/sglang/pull/38163) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-09-06 |
| labels | deepseek, jit-kernel, memory-pool |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回退 #30315:移除 use_req_ring 与 unified-KV 专属 SWA 记账,恢复原 KV 池容量核算。

## 问题现象
正文未描述具体现象,仅一句「Reverts sgl-project/sglang#30315」,附 3 条失败的 CI 状态(Latest PR Test Base / Extra / AMD ROCm 7.2,均为 :no_entry_sign:)。标题指向 AMD 平台 DSV4(DeepSeek V4)的 unified-KV 池容量与 SWA ring 记账问题。

## 根因
正文未披露根因(只给出被回退的 PR)。本次回退所撤销的机制(由 diff 推断):#30315 为 unified-KV 的 c4 压缩引入 per-request ring 路径(`use_req_ring = compress_ratio == 4 and token_to_kv_pool._unified_kv`),贯穿 jit/triton plan kernel 与 compressor/compressor_v2/compress_hip 全链路;并把调度侧 SWA 记账改为 ring 容量口径——`rem_swa_tokens` 不再加 `tree_cache.swa_evictable_size()`、`_swa_budget_for_req` 只按固定 ring slot 成本预算、chunked 续传免扣 SWA、`add_one_req` 准入判断由 `>=` 放宽为 `>`,另新增 `[SWA-BOTTLENECK]` 诊断日志和 `_pre_alloc` 中的 `clear_c4_req_states` 调用。

## 修复方案
整体 revert #30315,涉及 9 个文件:`kernels/jit/csrc/deepseek_v4/c_plan.cuh`(各 Params 删 `use_req_ring` 字段、删 `RuntimeCheck(!use_req_ring || compress_ratio == 4)`,`compute_req_ring_loc` 改回 `compute_c128_loc`)、`kernels/ops/attention/dsv4/attn.py`、`dsv4/compress.py`、`layers/attention/dsv4/compressor.py`、`compressor_v2.py`、`compress_hip.py`(删统一 KV 下 ratio==4 的分支)、`disaggregation/decode.py`(删 `clear_c4_req_states`)、`managers/schedule_batch.py`(删 `_log_unified_swa_bottleneck` 与 `import time`)、`managers/schedule_policy.py`。关键摘录(恢复原 `rem_swa_tokens`):

```diff
-        allocator = self.token_to_kv_pool_allocator
-        if getattr(allocator.get_kvcache(), "_unified_kv", False):
-            return allocator.swa_available_size() - self.rem_swa_token_offset
         return (
-            allocator.swa_available_size()
+            self.token_to_kv_pool_allocator.swa_available_size()
             + self.tree_cache.swa_evictable_size()
             - self.rem_swa_token_offset
         )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:按 diff,触及 DeepSeek V4 压缩注意力全链路(jit/triton plan kernel、compressor 三处)、PD 分离 decode 的 `_pre_alloc`、scheduler 的 SWA 预算与准入(`add_one_req` 恢复为 `>=`、chunked prefill 恢复 SWA clamp);unified-KV 下 c4 压缩回到 #30315 之前的行为。pack 的 diff 在 40000 字符处截断(按文件粒度),可能存在未展示的改动。

## 关联
- sgl-project/sglang#30315(正文唯一指向的被回退 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
