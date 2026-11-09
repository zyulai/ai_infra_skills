# PR #1326: Fix the flaky test test_moe_eval_accuracy_large.py

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1326](https://github.com/sgl-project/sglang/pull/1326) |
| 作者 | merrymercy |
| 合并日期 | 2024-09-04 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

`test_moe_eval_accuracy_large` 的 MGSM 英文评测存在波动,断言下限从 0.64 放宽到 0.63 以消除 flaky。

## 问题现象

`test/srt/test_moe_eval_accuracy_large.py` 中的 `test_mgsm_en` 评测不稳定(flaky):MoE 模型在 MGSM 英文评测上的 score 会偶尔低于原断言下限 0.64,导致测试间歇性失败。标题直接指出该测试为 flaky test;正文未进一步描述具体现象。

## 根因

正文为空,未披露根因。从 diff 看,唯一改动是把 `assert metrics["score"] >= 0.64` 放宽为 `>= 0.63`,即认可该评测分数存在跨运行波动,0.64 的阈值卡得太紧,偶发分数落在 0.63–0.64 区间时误报失败(由 diff 推断)。

## 修复方案

只改 `test/srt/test_moe_eval_accuracy_large.py` 中 `test_mgsm_en` 的断言阈值:

```diff
-        assert metrics["score"] >= 0.64, f"{metrics}"
+        assert metrics["score"] >= 0.63, f"{metrics}"
```

不涉及被测的 MoE 推理代码本身。

## 影响与回归风险

- 影响窗口:仅影响 CI 测试,不影响任何版本运行时行为。
- 回归面:`test_mgsm_en` 一条断言。放宽 0.01 后,真实精度回退到 0.63 以下仍会被捕获,但 0.63–0.64 之间的轻微回退将不再报警;这是以轻微灵敏度换取 CI 稳定性的常见取舍。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
