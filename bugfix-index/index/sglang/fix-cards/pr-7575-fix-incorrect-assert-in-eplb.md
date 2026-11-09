# PR #7575: [Fix] incorrect assert in EPLB

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7575](https://github.com/sgl-project/sglang/pull/7575) |
| 作者 | ch-wan |
| 合并日期 | 2025-06-26 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

EPLB 专家统计里针对多 token prediction 的断言本身写错,合法输入也会触发断言失败,直接删除该断言。

## 问题现象

启用 EPLB(专家并行负载均衡)的专家分布统计时,`on_select_experts` 中的断言会错误地失败。断言自身的信息称 "Selecting expert is not supported for multiple token prediction at the moment",即多 token prediction 场景下该路径会因形状不匹配的断言而崩溃,而实际统计逻辑本身可以正常处理。正文 Motivation 与 Modifications 均为空,未描述具体触发日志。

## 根因

正文未披露根因(Motivation/Modifications 留空)。从 diff 看,该断言比较 `self._data[layer_idx, :]` 的形状与 flatten 后的 `topk_ids` 形状;而紧随其后的 `scatter_add_` 只要求 index 与 src 形状一致(二者都来自 `topk_ids`),目标行的形状本就不必与 `topk_ids` 相同,因此该断言的前提就是错的,在多 token prediction 等输入形状不同的合法场景下误报(由 diff 推断)。

## 修复方案

改 `python/sglang/srt/managers/expert_distribution.py` 的 `on_select_experts`,删除该断言,保留原有统计逻辑:

```diff
         topk_ids = topk_ids.flatten()
         mask = topk_ids != -1
-        assert self._data[layer_idx, :].shape == topk_ids.shape, (
-            "Shape mismatch between data and topk_ids."
-            "Selecting expert is not supported for multiple token prediction at the moment."
-        )
         self._data[layer_idx, :].scatter_add_(
             dim=0, index=topk_ids.masked_fill(~mask, 0).long(), src=mask.int()
         )
```

无效专家(-1)仍通过 `masked_fill` 归零处理,统计行为不变。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;影响启用 EPLB 专家分布统计的路径,尤其是多 token prediction 输入。
- 回归面:仅移除一条断言,`scatter_add_` 统计逻辑未动。风险是失去该处的形状自检,但断言本身条件不成立,删除后不会改变任何合法输入的行为。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
