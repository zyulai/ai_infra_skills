# PR #14777: Fix VLM accuracy thresholds for nightly tests

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14777](https://github.com/sgl-project/sglang/pull/14777) |
| 作者 | alisonshao |
| 合并日期 | 2025-12-10 |
| labels | Multi-modal |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
两个 VLM 的 nightly MMMU 精度阈值定得比实际值高导致测试挂;按实测值下调阈值。

## 问题现象
nightly VLM 精度测试(`test/nightly/test_vlms_mmmu_eval.py`,基于 100 条 MMMU 样本)中,两个模型的阈值高于实际测量值,导致测试不通过:deepseek-ai/deepseek-vl2-small 阈值 0.33、实际值 0.3298;google/gemma-3n-E4B-it 阈值 0.36、实际值 0.2703。正文未描述模型推理本身存在精度回归。

## 根因
正文未披露根因,只给出阈值与实际值的对照(阈值定得比实测精度高)。

改动机制(由 diff 推断):`MODEL_THRESHOLDS` 表中的精度下限按各模型实测值放宽——deepseek-vl2-small 0.330→0.320(实际 0.3298),gemma-3n-E4B-it 0.360→0.270(实际 0.2703),使阈值略低于实际测量精度,nightly 测试得以通过。

## 修复方案
改动仅涉及 `test/nightly/test_vlms_mmmu_eval.py` 的 `MODEL_THRESHOLDS` 配置表,调整两个模型的精度阈值(延迟阈值不变):

```diff
     ModelLaunchSettings("deepseek-ai/deepseek-vl2-small"): ModelEvalMetrics(
-        0.330, 56.1
+        0.320, 56.1
     ),
-    ModelLaunchSettings("google/gemma-3n-E4B-it"): ModelEvalMetrics(0.360, 17.7),
+    ModelLaunchSettings("google/gemma-3n-E4B-it"): ModelEvalMetrics(0.270, 17.7),
```

Test plan:验证 nightly VLM 精度测试在更新阈值后通过。

## 影响与回归风险
- 影响窗口:不适用(纯测试配置改动,不涉及运行时代码)。
- 回归面:仅影响 nightly MMMU VLM 测试对 deepseek-vl2-small 与 gemma-3n-E4B-it 的判定;阈值放宽意味着若这两个模型未来真的出现精度回退(分别跌破 0.32 / 0.27),只有下探到新阈值才会报警,灵敏度略降。其余模型的阈值与延迟阈值均未改动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
