# PR #29023: Revert "[CI] Fix flaky optimistic test by adding contention handling"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29023](https://github.com/sgl-project/sglang/pull/29023) |
| 作者 | ShangmingCai |
| 合并日期 | 2026-06-23 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#28995 已修根因,#28947 为 flaky 测试加的 bootstrap 争用负载手段被整体回退。

## 问题现象
正文未描述具体现象,仅说明回退 #28947、因为 #28995 已修复根因。(由 diff 推断)被回退的是 PD disaggregation optimistic prefill 测试的「制造争用」辅助手段:测试时用 4 个并发请求占住 decode bootstrap(optimistic prefill 的 retry 只在 bootstrap 仍 pending 时发生,单个空闲请求会瞬间完成 bootstrap),以保证 retry 真正被触发、测试不 flaky。

## 根因
正文披露:#28947 只是给 flaky 测试添加争用处理(症状级 workaround),而 #28995 修复了根因,因此不再需要争用手段,整体回退前者。

## 修复方案
整体 revert #28947,`test/registered/disaggregation/test_disaggregation_optimistic_prefill.py`:
- 删除 `BOOTSTRAP_CONTENTION_REQUESTS/WORKERS` 常量与 `_run_under_bootstrap_contention` 辅助函数(其用 ThreadPoolExecutor 提交 4 个 400 词生成请求制造 decode bootstrap 争用);
- `test_logprob` 恢复为直接 POST `/generate` 并断言 retry 计数增加,不再包裹争用负载层。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响 test_disaggregation_optimistic_prefill.py 这一个测试文件;若 #28995 的根因修复有遗漏,该测试的 flaky 现象可能在 CI 上重新出现(按正文逻辑,#28995 已覆盖根因)。

## 关联
- sgl-project/sglang#28947(被回退的 PR)
- sgl-project/sglang#28995(根因修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
