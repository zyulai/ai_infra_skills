# PR #12866: [Fix] Fix nan error for large scale ep

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12866](https://github.com/sgl-project/sglang/pull/12866) |
| 作者 | Fridge003 |
| 合并日期 | 2025-11-11 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

#10874 误删首/尾节点冗余专家映射,致 w13_input_scale 全 NaN 并触发断言;改为保留冗余专家修复。

## 问题现象

大规模 EP(如 GB200,最新 main)下,第一个和最后一个节点上 `w13_input_scale` 全部变成 NaN:

```
[2025-11-08 02:52:56 DP2 TP2 EP2] w13_input_scale values: tensor([nan, nan, nan, nan, nan, nan], ...)
[2025-11-08 02:52:56 DP1 TP1 EP1] w13_input_scale values: tensor([nan, nan, nan, nan, nan, nan], ...)
```

随后在 `modelopt_quant.py:1460` 的 `process_weights_after_loading` 触发 `assert torch.all(w13_input_scale == w13_input_scale[0])` 报错。复现脚本见正文 gist 链接(GB200 + main)。

## 根因

正文披露:该 bug 由 #10874 引入——`logical_to_all_physical_map` 中冗余专家被错误移除,例如逻辑专家 0 应映射为 `[0, 256]`,却被错误地设为 `[0, -1]`;只发生在第一个和最后一个节点。正文修改说明为「保留这些节点上的冗余专家」。(由 diff 推断)具体机制:`_compute_logical_to_all_physical_map` 中「替换为最近物理专家」的分支原先无条件把 `mapped_physical_experts[0]` 写成 `nearest_expert`,当 nearest_expert 不在映射列表内时覆盖了原有的有效条目、留下错误映射,导致这些节点权重加载异常并产生 NaN。

## 修复方案

改 `python/sglang/srt/eplb/expert_location.py` 的 `_compute_logical_to_all_physical_map`:替换最近物理专家前先做去重判断,只有 `nearest_expert` 有效且**不在**当前映射列表中时才写 `mapped_physical_experts[0]`,不再无条件覆盖:

```diff
-                if nearest_expert != -1:
-                    logical_to_all_physical_map[layer_id][logical_expert_id] = [
-                        nearest_expert
-                    ]
+                mapped_physical_experts = logical_to_all_physical_map[layer_id][
+                    logical_expert_id
+                ]
+                if (
+                    nearest_expert != -1
+                    and nearest_expert not in mapped_physical_experts
+                ):
+                    mapped_physical_experts[0] = nearest_expert
```

即首/尾节点上的冗余专家映射得以保留,`[0, 256]` 这类映射不再被写成 `[0, -1]`。

## 影响与回归风险

- 回归面:EPLB 专家位置映射(`_compute_logical_to_all_physical_map`)中「nearest expert 替换」分支——影响 EP 部署下逻辑到物理专家映射的构建;仅当 nearest_expert 已存在于映射列表时行为改变(不再覆盖),其他场景不变。作者注明该修复使问题得到解决;未附精度/性能基准。

## 关联

- #12293(所属 issue)
- #10874(引入该 bug 的 PR)
- 复现脚本:[gist (kaixih)](https://gist.github.com/kaixih/32bdc4fec4feabe9305d1acb2e1f96db)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
