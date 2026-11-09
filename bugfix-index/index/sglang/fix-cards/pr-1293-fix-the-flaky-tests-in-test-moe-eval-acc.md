# PR #1293: Fix the flaky tests in test_moe_eval_accuracy_large.py

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1293](https://github.com/sgl-project/sglang/pull/1293) |
| 作者 | merrymercy |
| 合并日期 | 2024-09-01 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`test_moe_eval_accuracy_large.py` 的 MoE 精度评测偶发失败,把 `test_human_eval` 的分数断言阈值从 0.43 放宽到 0.42 消除 flaky。

## 问题现象
正文未描述具体现象。从标题与 diff 看,涉及 `test/srt/test_moe_eval_accuracy_large.py` 中 MoE 大规模精度评测测试的偶发(flaky)失败;diff 中被改动的是 `test_human_eval` 的分数下限断言。

## 根因
正文未披露根因。(由 diff 推断)`test_human_eval` 通过 `run_eval(args)` 拿到 `metrics` 后断言 `metrics["score"] >= 0.43`,而 MoE 评测分数在多次运行间存在波动,分数落在 0.43 以下时测试会偶发报错;作者将阈值下调 0.01 至 0.42,以容忍这种运行间波动。

## 修复方案
只改 `test/srt/test_moe_eval_accuracy_large.py` 一处断言:

```diff
         metrics = run_eval(args)
-        assert metrics["score"] >= 0.43, f"{metrics}"
+        assert metrics["score"] >= 0.42, f"{metrics}"
```

同文件中的其他测试(如 `test_mgsm_en`)未在 diff 中改动。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:纯测试文件改动,不触及任何运行时代码路径。风险在于断言容忍度放宽 0.01:若 MoE 实现出现使 `test_human_eval` 分数落在 0.42–0.43 区间的真实精度回归,该测试将不再拦截。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
