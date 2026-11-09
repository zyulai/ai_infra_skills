# PR #28190: fix(precision): do not promote failed runs to the comparison baseline

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28190](https://github.com/sgl-project/sglang/pull/28190) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-07-02 |
| labels | high priority, run-ci, bypass-fastfail |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
nightly 精度测试按 push_index 选基线、无视 pass_label,failed(回退)run 被提升为次日基线,一次性掩盖持续性回退;改为优先非 failed 行、仅在全 failed 时回落。

## 问题现象
nightly precision regression test(承接 #26902)出现:N 夜引入回退当晚 CI 变红,N+1 起一直绿,未修复的回退看起来像"自愈"。时序:N-1 好基线 PASS;N 出现回退,today(regressed) vs good baseline FAIL,回退张量以 `pass_label="failed"` 上传(本意只是保留诊断 diff);N+1 基线即 N 的 failed run,today(still regressed) vs regressed 变 PASS;N+2 起永远绿。发生在名义的 HF 驱动 / cross-runner 模式(即 HF store 面向的冷本地基线场景)。区别于已记录的 "rolling baseline" 渐进漂移局限:这里是尖锐的单夜回退/持续性回退被一步吸收。

## 根因
正文披露:`_select_latest_run` 纯按 push_index(新旧)选最新行、忽略 `pass_label`;FAILED 分支上传的 failed run 因此成为次夜参考基线,掩盖持续性回退,也与 FAILED 分支注释声明的意图("so the diagnostic diff survives on HF")矛盾。

## 修复方案
`python/sglang/test/precision_baseline_store.py` 的 `_select_latest_run` 排序后改为优先最新非 failed 行:

```diff
-    return candidates[-1][1]["run_path"]
+    usable = [c for c in candidates if c[1].get("pass_label") != "failed"]
+    chosen = usable or candidates
+    return chosen[-1][1]["run_path"]
```

缺 `pass_label` 的 legacy 行视为可用(向后兼容)。新增 4 个单测:`test_prefers_older_passed_over_newer_failed`、`test_prefers_baseline_established_over_newer_failed`、`test_falls_back_to_failed_when_only_failed`、`test_missing_pass_label_treated_as_usable`;既有 TestSelectLatestRun 用例全通过(无 failed 行时行为不变)。

## 影响与回归风险
- 影响窗口:正文未说明(该选择逻辑自 #26902 引入 nightly 精度测试起存在)。
- 回归面:仅 `fetch_latest_baseline` 的基线选择;无 failed 行时行为不变;全部 failed 时仍选最新 failed(保持可用)。修复后持续性回退每晚都被抓到,修复合入后基线前进到最新 passed run。

## 关联
- Follow-up to:#26902

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
