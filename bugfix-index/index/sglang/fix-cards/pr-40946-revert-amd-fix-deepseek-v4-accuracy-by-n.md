# PR #40946: Revert "[AMD] Fix DeepSeek-V4 accuracy by not passing num_token_non_padded to MoE topk"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#40946](https://github.com/sgl-project/sglang/pull/40946) |
| 作者 | At1a8 |
| 合并日期 | 2026-09-24 |
| labels | deepseek |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
回滚 #39804 的 AMD DeepSeek-V4 精度 workaround:#40672 已更早修复 padding 精度问题,GSM8K 全配置验证持平。

## 问题现象
正文未描述用户侧症状(被回滚的 #39804 针对的 DeepSeek-V4 精度问题已由 #40672 解决)。本 PR 的「现象」是代码层面:#39804 在 AMD(HIP)上把 MoE topk 的 `num_token_non_padded` 强制置 None 的 workaround 成为冗余分支,需要移除。回滚验证:GSM8K(1319 题)精度检查通过,8 种 tbo/spec/并行组合下 main(含 #39804)与本 PR(回滚 #39804)精度均在 0.943-0.954 区间、基本持平。

## 根因
正文披露:#39804(「[AMD] Fix DeepSeek-V4 accuracy by not passing num_token_non_padded to MoE topk」)为解决 AMD 上 padding 引起的精度问题而绕过 `num_token_non_padded`,但 PR #40672 已经修复了 padding 造成的精度问题且合并时间更早,#39804 的修改因此成为多余且偏离主路径的特例,故整体回滚。

## 修复方案
`python/sglang/srt/models/deepseek_v2.py` `forward`:删除 `_is_hip` 分支——

```diff
-        if _is_hip:
-            num_token_non_padded = None
-        else:
-            num_token_non_padded = (
-                forward_batch.moe_num_token_non_padded()
-                if forward_batch is not None
-                else None
-            )
+        num_token_non_padded = (
+            forward_batch.moe_num_token_non_padded()
+            if forward_batch is not None
+            else None
+        )
```

即恢复为:无论硬件平台,`num_token_non_padded` 一律按 `forward_batch.moe_num_token_non_padded()` 取值传入 MoE topk,不再在 AMD/ROCm 上强制置 None。

## 影响与回归风险
- 影响窗口:#39804 合并之后至本回滚之前。
- 回归面:AMD(HIP)上 DeepSeek-V2/V4 系列 MoE 路由重新接收 `num_token_non_padded`(与 CUDA 行为一致);由于 #40672 已修复 padding 精度问题,预期无精度回归——正文以 GSM8K 8 配置矩阵(DP/EP、MTP、TBO 组合,0.943-0.954)验证;若存在 #40672 未覆盖的 AMD 场景,可能重新暴露 padding 精度问题(正文未提及此类场景)。

## 关联
- Reverts: https://github.com/sgl-project/sglang/pull/39804
- 前置修复: https://github.com/sgl-project/sglang/pull/40672

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
