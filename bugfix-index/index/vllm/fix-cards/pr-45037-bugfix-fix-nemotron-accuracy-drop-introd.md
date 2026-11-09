# PR #45037: [Bugfix] Fix nemotron accuracy drop introduced by #41184

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45037](https://github.com/vllm-project/vllm/pull/45037) |
| 作者 | bnellnm |
| 合并日期 | 2026-06-10 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
#41184 之后 Nemotron 的 `routed_scaling_factor` 被应用了两次导致精度下降;`apply_routed_scale_to_output` 为真时改为传 1.0 给 RoutedExperts。

## 问题现象
Nemotron 模型(如 Nemotron-3-Super-120B-A12B-NVFP4)出现精度下降。正文未给出具体的精度数值对比或日志,仅说明错误为 `routed_scaling_factor` 被重复应用;用 gsm8k correctness 测试验证,修复后通过。

## 根因
PR 正文披露:#41184 之后,routed scaling factor 在 Nemotron 模型中被应用了两次——外层已经按 `apply_routed_scaling_factor_to_output` 把缩放因子应用到输出,同时又将该缩放因子原值传给了 `RoutedExperts`,导致专家内部再次缩放。因此需要在把 scaling factor 传给 `RoutedExperts` 之前检查 `apply_routed_scaling_factor_to_output`。

## 修复方案
改 `vllm/model_executor/layers/fused_moe/layer.py` 中 `FusedMoE` 的构造传参:当 `apply_routed_scale_to_output` 为真时,传给内层 `RoutedExperts` 的 `routed_scaling_factor` 用 1.0,否则传原值——保证缩放只发生一次。

```diff
-        routed_scaling_factor=routed_scaling_factor,
+        routed_scaling_factor=routed_scaling_factor
+        if not apply_routed_scale_to_output
+        else 1.0,
```

## 影响与回归风险
- 影响窗口:自 #41184 引入起(标题与正文均指向该 PR)。
- 回归面:所有经 `FusedMoE` wrapper 且 `apply_routed_scale_to_output` 为真的模型构建路径(不止 Nemotron):这些模型的专家内层不再应用缩放,依赖外层应用;若某模型此前依赖内层应用缩放而外层 flag 被误置真,输出会变为未缩放。`apply_routed_scale_to_output` 为假的路径行为不变。

## 关联
- PR #41184(引入精度回退的前置改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
