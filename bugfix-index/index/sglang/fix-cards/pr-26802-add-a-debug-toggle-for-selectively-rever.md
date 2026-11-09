# PR #26802: Add a debug toggle for selectively reverting PR fixes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26802](https://github.com/sgl-project/sglang/pull/26802) |
| 作者 | fzyzcjy |
| 合并日期 | 2026-05-31 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

新增 `pr_fix_toggle` 调试工具:经 `SGLANG_DEBUG_REVERT_PR` 环境变量在运行时用源码补丁器反向套用指定 PR 的修复,便于回归 A/B 调试。

## 问题现象

正文未描述具体现象。本 PR 不是修某个具体 bug,而是引入回归调试工具:模块 docstring 写明用途为 "Reverse-apply historical PR fixes for regression-style tests"(为回归类测试反向套用历史 PR 修复)。工具内注册了两个可回滚对象:PR 25015 与 PR 26329。

## 根因

正文未披露根因(本 PR 为调试基础设施,不针对单一 bug)。从实现看,其动机是当某个已合并的 PR 修复疑似引入回归时,可在不改代码的情况下按 PR 号在运行时撤销该修复做 A/B 验证。(由 diff 推断)

## 修复方案

四处改动:

1. 新增 `python/sglang/srt/debug_utils/pr_fix_toggle.py`:`maybe_revert_pr_fix()` 入口,读取新环境变量 `SGLANG_DEBUG_REVERT_PR`(EnvInt,默认 0);内置 `_PR_FIX_REVERT_YAML` 注册表,含 25015(针对 `eagle_worker.EAGLEWorker.draft_forward`、`eagle_worker_v2.EagleDraftWorker.draft_forward`、`EAGLEDraftCudaGraphRunner.capture_one_batch_size` 的 match/replacement 补丁)与 26329(针对 `eagle_utils._eagle_prefill_tail_tokens` 加 `return` 前置)两份 YAML;未注册的 PR 号抛 `NotImplementedError`。
2. `Scheduler.__init__` 在 `init_batch_result_processor()` 之后调用 `maybe_revert_pr_fix()`。
3. 增强 `source_editor.py` 的 `_find_match` 失败路径:新增 `_not_found_diagnostic`,报告源码行数,并在 match 首行存在时打印最多 8 个带 `>` 标记的上下文窗口,便于定位漂移的 match 文本。
4. `test_source_editor.py` 新增对应诊断行为用例(source_len、无命中、单/多窗口、8 窗口上限、边界 clamp、多行 match 标记)。

## 影响与回归风险

- 影响窗口:不适用(新增工具,非修复窗口)。
- 回归面:默认路径无行为变化——`SGLANG_DEBUG_REVERT_PR` 默认 0,`maybe_revert_pr_fix()` 为空操作;仅当显式设置该环境变量时才在 Scheduler 初始化阶段对 eagle 相关源码打反向补丁(影响投机解码路径,属调试行为)。`_find_match` 报错文案变化仅影响补丁失败时的诊断输出。

## 关联

正文未列出链接(代码注册表中的可回滚对象:PR 25015、PR 26329)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
