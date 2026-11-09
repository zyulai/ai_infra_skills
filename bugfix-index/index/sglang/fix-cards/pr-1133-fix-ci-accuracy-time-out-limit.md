# PR #1133: Fix CI accuracy && time out limit

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1133](https://github.com/sgl-project/sglang/pull/1133) |
| 作者 | hnyls2002 |
| 合并日期 | 2024-08-17 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
CI 校准:新增 mixed chunked prefill 后 backend 单测超出 15 分钟超时,MMLU 分数低于 0.71 断言;超时提到 18 分钟、阈值降到 0.705。

## 问题现象
正文未描述具体报错或日志。从动机可读出两类 CI 问题(现象层面为推断):backend runtime 单测套件在新增 mixed chunked prefill 测试后总时长超过工作流 15 分钟的 `timeout-minutes` 限制而失败;MMLU 大规模精度评测的实际分数落到 0.705–0.71 区间,使 `assert metrics["score"] >= 0.71` 失败。

## 根因
正文未披露根因。Motivation 仅说明动因:此前加入了 mixed chunked prefill(测试),因此把 backend runtime 单测超时上限从 15 分钟提高到 18 分钟;同时把 MMLU 精度断言从 0.71 下调到 0.705。正文没有解释 MMLU 分数为何低于 0.71(是模型/后端真实变化还是评测波动未说明),也未说明下调阈值是否掩盖了真实回归。

## 修复方案
纯 CI 配置与测试断言改动,不触及运行时代码:
- `.github/workflows/unit-test.yml`:minimal 套件步骤 `timeout-minutes: 15` → `18`。
- 三个评测测试的 MMLU 断言同步下调 `0.71` → `0.705`:`test/srt/test_eval_accuracy_large.py`、`test/srt/test_eval_accuracy_large_chunked_prefill.py`、`test/srt/test_eval_accuracy_large_mixed_chunked_prefill.py`。

关键 diff:

```diff
-        assert metrics["score"] >= 0.71, f"{metrics}"
+        assert metrics["score"] >= 0.705, f"{metrics}"
```

## 影响与回归风险
- 影响窗口:正文未提及(改动仅针对当时 CI)。
- 回归面:仅影响 GitHub Actions 工作流与三个 MMLU 评测测试的通过标准——超时上限放宽、精度门槛降低 0.005;不改变推理引擎任何代码路径,无线上行为风险。需要注意的是,下调断言阈值意味着 0.705–0.71 之间的真实精度劣化不会再被该测试捕获(按 diff 如实陈述)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
