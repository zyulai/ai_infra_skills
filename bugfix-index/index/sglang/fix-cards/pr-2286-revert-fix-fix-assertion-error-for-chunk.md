# PR #2286: Revert "[Fix] fix assertion error for chunked prefill when disabling cache"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2286](https://github.com/sgl-project/sglang/pull/2286) |
| 作者 | merrymercy |
| 合并日期 | 2024-12-01 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论
回退 #2282:删除调度策略中"can_run_list 非空即返回 OTHER"的提前返回,恢复原有 chunked prefill 调度流程。

## 问题现象
正文仅一句 "Reverts sgl-project/sglang#2282",未描述具体现象。被回退的原 PR 标题为 "[Fix] fix assertion error for chunked prefill when disabling cache",本 PR 将其撤销;原 PR 针对的现象(禁用 cache 时 chunked prefill 触发 assertion error)与本回退自身的症状均未在正文详述。

## 根因
正文未披露根因,仅说明是对 #2282 的 revert。(由 diff 推断)机制:#2282 曾在 `schedule_policy.py` 的调度逻辑中加入提前返回——只要 `self.can_run_list` 非空就 `return AddReqResult.OTHER`,以此绕开禁用 cache 时 chunked prefill 的断言错误;本次回退删除这两行,使调度继续走原有的 `req_states` 状态记录与后续判断流程。即该提前返回的修法被判定带来回归(如打断正常调度路径),整体撤销。

## 修复方案
改动文件 `python/sglang/srt/managers/schedule_policy.py`:在 `add_req_state` 之后、`if self.req_states is None:` 分支之前,删除:

```diff
-        if len(self.can_run_list) != 0:
-            return AddReqResult.OTHER
```

其余调度逻辑(`req_states` 插入排序等)保持不变。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,即 #2282 合入后至本回退落地之间。
- 回归面:`schedule_policy.py` 的请求调度/准入路径(_prefill 预算相关策略)。删除提前返回后,凡此前依赖该路径提前结束调度的场景会重新走完整流程;禁用 cache + chunked prefill 的原始 assertion error 场景在回退后回到 #2282 之前的行为,可能再次暴露,需以其他方式修复。

## 关联
- [sgl-project/sglang#2282](https://github.com/sgl-project/sglang/pull/2282)(正文注明的被 revert 对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
