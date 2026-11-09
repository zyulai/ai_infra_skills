# PR #17099: Fix hybrid attention PD Disaggregation test

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17099](https://github.com/sgl-project/sglang/pull/17099) |
| 作者 | ShangmingCai |
| 合并日期 | 2026-01-15 |
| labels | high priority |
| 热度 | 👍 0 · 💬 30 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
hybrid attention PD 分离测试在 CI 中不稳定(flaky),先在 CI 中临时跳过该测试并把套件超时从 600 秒降到 400 秒。

## 问题现象
正文未描述具体现象(Motivation/Modifications 均为模板占位符,没有填写)。从 diff 可见被改的是 `test/srt/test_disaggregation_hybrid_attention.py` 中的 `TestDisaggregationHybridAttentionMamba`(hybrid attention + Mamba 的 PD 分离服务测试)以及 `run_suite.py` 中 `per-commit-8-gpu-h200` 套件里该测试的超时配置。

## 根因
正文未披露根因。

(由 diff 推断)该测试在 CI 环境中属于不稳定(flaky)测试:diff 在测试类上加上了 `@unittest.skipIf(is_in_ci(), "Temporarily disable the flaky test.")`,即作者确认其在 CI 中会间歇性失败,故采取「CI 中临时禁用」的处理;同时把 `run_suite.py` 中 `per-commit-8-gpu-h200` 套件里该测试的时长配额从 600 秒调整为 400 秒。测试本体与被测的 PD 分离功能代码均未改动,本次修复仅针对 CI 测试编排。

## 修复方案
改了两个文件:

1. `test/srt/test_disaggregation_hybrid_attention.py`:从 `sglang.test.test_utils` 增加 `is_in_ci` 导入,并在 `TestDisaggregationHybridAttentionMamba` 类上添加 CI 跳过装饰器。
2. `test/srt/run_suite.py`:将 `per-commit-8-gpu-h200` 套件中 `test_disaggregation_hybrid_attention.py` 的时长从 600 改为 400。

关键 diff 摘录:

```diff
+    is_in_ci,
     popen_launch_pd_server,
 )
 
 
+@unittest.skipIf(is_in_ci(), "Temporarily disable the flaky test.")
 class TestDisaggregationHybridAttentionMamba(PDDisaggregationServerBase):
```

```diff
-        TestFile("test_disaggregation_hybrid_attention.py", 600),
+        TestFile("test_disaggregation_hybrid_attention.py", 400),
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 CI 测试执行——该 PD 分离测试在 CI 中不再运行(本地非 CI 环境仍会执行),`per-commit-8-gpu-h200` 套件中该测试的失败不再阻塞流水线;不涉及任何生产代码路径。风险是真实的 PD 分离回归可能因测试被跳过而在 CI 中漏检,需待测试稳定后恢复。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
