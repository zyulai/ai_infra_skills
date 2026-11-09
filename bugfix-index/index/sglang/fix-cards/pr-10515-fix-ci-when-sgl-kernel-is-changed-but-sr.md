# PR #10515: Fix CI when sgl-kernel is changed but srt is not changed

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10515](https://github.com/sgl-project/sglang/pull/10515) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-09-16 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
sgl-kernel 变更而 srt 未变时 CI 测试 job 被跳过;job 条件补上 sgl_kernel 变更触发分支。

## 问题现象
正文未描述具体现象(Motivation/Modifications 等章节均为空模板)。从标题可知问题为:当 PR 只改了 sgl-kernel 而未改 srt 时,CI 的相关测试 job 未按预期执行。

## 根因
正文未披露根因(Motivation 为空模板)。从 diff 看,机制是 `.github/workflows/pr-test.yml` 中 13 个测试/性能/精度 job 的 `if` 条件只检查 `needs.check-changes.outputs.main_package == 'true'`,即仅当 srt(main_package)有变更时才运行;sgl-kernel 变更时 `main_package` 输出为 false,这些 job 被整体跳过,sgl-kernel 的改动缺少对应测试覆盖(由 diff 推断)。

## 修复方案
改 `.github/workflows/pr-test.yml`,把下列 job 的运行条件从单一 `main_package` 判断扩为「main_package 或 sgl_kernel 任一变更即运行」:
`unit-test-frontend`、`unit-test-backend-1-gpu`、`unit-test-backend-2-gpu`、`unit-test-backend-4-gpu`、`unit-test-backend-8-gpu`、`unit-test-backend-8-gpu-b200`、`performance-test-1-gpu-part-1`、`performance-test-1-gpu-part-2`、`performance-test-2-gpu`、`accuracy-test-1-gpu`、`accuracy-test-2-gpu`、`unit-test-deepep-4-gpu`、`unit-test-deepep-8-gpu` 共 13 个 job,统一改为:
```diff
-        needs.check-changes.outputs.main_package == 'true'
+        ((needs.check-changes.outputs.main_package == 'true') || (needs.check-changes.outputs.sgl_kernel == 'true'))
```
各 job 的 `needs` 依赖(`check-changes`、`sgl-kernel-build-wheels` 及上游测试 job)保持不变。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 CI 工作流触发条件,不影响任何运行时代码。行为变化:只改 sgl-kernel 的 PR 现在会额外运行这 13 个 GPU 测试/性能/精度 job,CI 资源消耗增加;若 `check-changes` 的 `sgl_kernel` 输出判定过宽,可能让无关 PR 跑更多 job。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
