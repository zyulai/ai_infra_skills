# PR #15004: Fix CI by reverting incorrect metric check logic

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15004](https://github.com/sgl-project/sglang/pull/15004) |
| 作者 | Kangyan-Zhou |
| 合并日期 | 2025-12-12 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

CI 新加的"按输出模式智能重试"判断不正确导致 CI 出问题;整体移除该重试机制并恢复原有超时与直接失败语义。

## 问题现象

PR 正文只给出一条信息:该机制已在 CI job 中造成问题(fail 的 job 链接:[runs/20174414241/job/57919651680](https://github.com/sgl-project/sglang/actions/runs/20174414241/job/57919651680)),未描述具体错误输出。受影响面为 PR 测试工作流中被加上 `--enable-retry` 的各个 suite(quantization、per-commit 1/2/4/8-gpu、deepep、b200)以及 accuracy 测试的 `SGLANG_CI_ENABLE_RETRY` 环境变量路径。

## 根因

正文未披露根因细节,仅说明"incorrect metric check logic"造成了 CI 问题并整体 revert。(由 diff 推断)被 revert 的是此前加入的智能重试机制:`ci_utils.py` 里用 `RETRIABLE_PATTERNS`(匹配 `accuracy`、`score`、`latency`、`throughput` 等宽泛词)与 `NON_RETRIABLE_PATTERNS`(`RuntimeError`、`OOM` 等)对测试输出做正则判断,未匹配的 `AssertionError` 一律视为可重试;这种基于文本模式区分"真实代码错误"与"指标抖动"的判断不可靠,导致失败测试被反复重试、CI 行为异常。

## 修复方案

- `python/sglang/test/ci/ci_utils.py`:删除 `RETRIABLE_PATTERNS`、`NON_RETRIABLE_PATTERNS`、`is_retriable_failure`、`write_github_step_summary` 及重试循环,`run_unittest_files` 恢复为无重试、首败即停(或 continue_on_error 时继续)的原始逻辑;
- `test/run_suite.py` 与 `test/srt/run_suite.py`:删除 `--enable-retry`、`--max-attempts`、`--retry-wait-seconds` 参数及透传;
- `.github/workflows/pr-test.yml`:去掉各 suite 命令中的 `--enable-retry` 与顶层 `SGLANG_CI_ENABLE_RETRY` 环境变量及 accuracy 测试的 env 透传,`timeout-minutes` 一并恢复(40→30、30→20、25→20、55→45 等)。

## 影响与回归风险

- 影响窗口:正文未说明(即引入智能重试机制至本 revert)。
- 回归面:仅 CI 基建(`pr-test.yml`、`ci_utils.py`、两处 `run_suite.py`),不影响产品代码路径。恢复后指标类断言失败不再自动重试,PR 测试直接以失败告终;若此前依赖重试掩盖的偶发抖动,将重新显式暴露。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
