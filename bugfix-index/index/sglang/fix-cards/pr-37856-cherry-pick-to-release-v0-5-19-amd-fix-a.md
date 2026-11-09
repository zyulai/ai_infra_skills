# PR #37856: [Cherry-pick to release/v0.5.19] [AMD] fix aiter cannot get heuristic kernel regression (#37438)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37856](https://github.com/sgl-project/sglang/pull/37856) |
| 作者 | Qiaolin-Yu |
| 合并日期 | 2026-09-03 |
| labels | cherry-pick |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
AMD aiter 后端零填充+FP8 场景误禁 PS kernel 导致回落错误内核,改为仅在选中 Gluon decode 时才禁用;cherry-pick 到 v0.5.19。

## 问题现象
正文为 cherry-pick 说明,未单独描述现象;按原始标题(#37438),这是 aiter「cannot get heuristic kernel」的性能回归。正文 CI 状态显示 Base、Extra、AMD ROCm 7.2 三个测试运行均为失败(:x:)(cherry-pick 为手工操作,bot 工作流当时无法推送分支)。

## 根因
正文未披露详细根因,仅由原始标题指明这是 aiter 无法取到 heuristic kernel 的回归。
(由 diff 推断)机制:在 `aiter_backend.py` 的 zero-pad 拓扑(h12->qh16)分支中,只要 `head_pad_mode == "zero"` 且 `kv_cache_dtype` 为 fp8,就无条件关闭 `_use_mla_ps_kernel`/`fast_mode`/`intra_batch_mode`;该设计意图是让零填充场景优先选 Gluon decode,但当条件不满足 Gluon 选择(如 `prefer_mla_gluon_decode` 判否)时,内核选择链已失去 PS 候选,只能回落到不正确的 aiter 内核,回归由此产生。修复注释原文:"Disable ps only when gluon kernel is selected to avoid falling back to incorrect aiter kernel"。

## 修复方案
改 `python/sglang/srt/layers/attention/aiter_backend.py`(cherry-pick 自 #37438 的 commit `44a92e5`):把无条件禁用改为先用 `prefer_mla_gluon_decode(...)` 判定,仅当确实选中 Gluon decode 时才禁用 PS:
```diff
-                _use_mla_ps_kernel = False
-                fast_mode = False
-                intra_batch_mode = False
+                # Disable ps only when gluon kernel is selected to avoid falling
+                # back to incorrect aiter kernel
+                if prefer_mla_gluon_decode(
+                    head_pad_mode=self.head_pad_mode,
+                    num_head=self.num_head,
+                    kv_cache_dtype=self.kv_cache_dtype,
+                ):
+                    _use_mla_ps_kernel = False
+                    fast_mode = False
+                    intra_batch_mode = False
```
`self.max_split_per_batch = 32 if _use_mla_ps_kernel else None` 与 `log_mla_gluon_capability` 调用保持不变。

## 影响与回归风险
- 影响窗口:正文未说明;本卡针对 release/v0.5.19 分支,回归源头为主线的引入提交(原始 PR #37438 所述)。
- 回归面:AMD aiter attention 后端的 MLA decode 内核选择路径,具体为 zero-pad 拓扑 + fp8 KV cache 下的 PS/Gluon/aiter 内核决策;未选 Gluon 的组合现在保留 PS kernel(`_use_mla_ps_kernel=True`,`max_split_per_batch=32`),选择行为与回归前对齐;`prefer_mla_gluon_decode` 依赖的入参(head_pad_mode/num_head/kv_cache_dtype)若与实际内核选择条件不一致,可能出现判断偏差。

## 关联
- https://github.com/sgl-project/sglang/pull/37438 (源 PR)
- CI 运行:https://github.com/sgl-project/sglang/actions/runs/33795260871 等(正文 CI 状态链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
