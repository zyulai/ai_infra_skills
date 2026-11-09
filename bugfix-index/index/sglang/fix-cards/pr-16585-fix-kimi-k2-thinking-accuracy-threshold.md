# PR #16585: fix: kimi k2 thinking accuracy threshold change

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16585](https://github.com/sgl-project/sglang/pull/16585) |
| 作者 | dougyster |
| 合并日期 | 2026-01-06 |
| labels | deepseek |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

Kimi K2 的 nightly GSM8K 准确率常以 0.945 低于 0.95 阈值而误报失败;把基线阈值从 0.95 降到 0.94 以减少抖动。

## 问题现象

nightly 中 Kimi K2 使用 GSM8K 做准确率测试,经常以 `0.945 < 0.95` 的形式失败——准确率本身稳定在阈值附近略低处,导致测试间歇性误报。触发条件:8-gpu nightly 跑 `test/registered/8-gpu-models/test_kimi_k2.py` 的 Kimi-K2-Thinking Unified 测试。影响面为该 nightly 测试的稳定性(flakiness),不是模型推理质量问题。

## 根因

正文披露:Kimi K2 的 GSM8K 准确率常年在 0.945 左右,而现有基线阈值设为 0.95,阈值定得过高导致频繁误报失败。修法是把阈值下调到 0.94 以降低抖动。

## 修复方案

改 `test/registered/8-gpu-models/test_kimi_k2.py`,把 `AccuracyTestParams` 的 `baseline_accuracy` 从 0.95 改为 0.94:

```diff
-            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.95),
+            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.94),
```

仅改 nightly 测试的判定阈值,不触碰任何产品代码。

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 `test_kimi_k2.py` 的准确率判定。阈值降低 0.01 意味着 0.94-0.95 区间的真实精度回归将不再被该测试拦截,存在漏报风险;对推理代码、kernel、模型输出均无影响。正文 Accuracy/Benchmarking 均标注 n/a。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
