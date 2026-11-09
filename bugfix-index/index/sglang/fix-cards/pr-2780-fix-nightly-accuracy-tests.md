# PR #2780: Fix nightly accuracy tests

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2780](https://github.com/sgl-project/sglang/pull/2780) |
| 作者 | merrymercy |
| 合并日期 | 2025-01-08 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

夜测 gsm8k 阈值逐模型调整并输出 GITHUB_STEP_SUMMARY 汇总,禁用总是超时的 human eval,移除 TP2 列表里重复的 DeepSeek-Coder-V2-Lite 模型。

## 问题现象

nightly 精度测试不可靠:gsm8k 夜测部分模型分数贴着阈值导致不稳定;human eval 总是超时(正文原文:it is always a timeout);TP2 夜测模型列表里 `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` 与 TP1 重复,同一模型被测两次。正文未给出具体失败日志。

## 根因

正文未系统披露根因,仅逐条给出动机:gsm8k 各模型的分数阈值与实际分数分布不再匹配,需调整(具体每个阈值的依据正文未说明,调整数值见 diff);human eval 每次运行都超时,故禁用;TP2 列表中的 DeepSeek-Coder-V2-Lite-Instruct 是重复条目(该模型已在 TP1 列表中),故移除。

## 修复方案

1. `test/srt/test_nightly_gsm8k_eval.py`:`MODEL_SCORE_THRESHOLDS` 12 个模型阈值调整——上调如 DeepSeek-Coder-V2-Lite-Instruct 0.84→0.85、Mixtral-8x7B 0.63→0.64、Qwen2-57B-A14B 0.87→0.88、FP8 变体普遍 +0.01;下调如 Llama-3.1-8B-Instruct 0.83→0.82、Llama-3.1-70B-Instruct 0.96→0.95、Meta-Llama-3.1-70B-FP8 0.95→0.94 等;`check_model_scores` 新增构建 markdown 汇总表并 `write_github_step_summary` 写入 `GITHUB_STEP_SUMMARY`;
2. `test/srt/run_suite.py`:nightly 套件移除 `test_nightly_human_eval.py`(该文件类改名为 `TestNightlyHumanEval` 保留);
3. `test_utils.py`:`DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2` 移除重复的 `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`;顺带清理 `launch_server` 的 DEVNULL 重定向与 `TestEvalAccuracyLarge`→`TestNightlyGsm8KEval` 改名、进程清理改到每模型循环内,`test_skip_tokenizer_init.py` 两处裸 assert 改为 `assertEqual`。

关键 diff 摘录(阈值示例):

```diff
-    "meta-llama/Llama-3.1-8B-Instruct": 0.83,
+    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
-    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.84,
+    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.85,
-    "meta-llama/Llama-3.1-70B-Instruct": 0.96,
+    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅测试基础设施——nightly gsm8k 通过标准松紧变化(12 个模型阈值)、CI 汇总输出新增、human eval 从 nightly 套件移除、TP2 夜测少跑一个模型;不涉及任何推理运行时代码。

## 关联

正文未列出(正文 @ 了 @zhyncs,无链接 issue)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
