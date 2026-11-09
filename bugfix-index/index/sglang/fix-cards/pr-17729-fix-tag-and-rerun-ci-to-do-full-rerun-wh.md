# PR #17729: Fix /tag-and-rerun-ci to do full rerun when PR has sgl-kernel changes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17729](https://github.com/sgl-project/sglang/pull/17729) |
| 作者 | alisonshao |
| 合并日期 | 2026-01-29 |
| labels | (none) |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
PR 含 sgl-kernel 改动时,`/tag-and-rerun-ci` 只重跑失败测试任务、不重跑内核构建,测试因拿不到 artifact 而失败;改为改用整跑重跑。

## 问题现象
当一个 PR 修改了 sgl-kernel 且测试失败时,用户执行 `/tag-and-rerun-ci` 后:
1. `rerun_failed_jobs()` 只重跑失败的测试 job,不重跑 `sgl-kernel-build-wheels` 构建任务;
2. 测试任务尝试下载 artifact,但 artifact 只归属于最初那次 workflow run;
3. 测试任务报错 "Found 0 artifact(s)" 失败。

正文给出的示例失败 run:https://github.com/sgl-project/sglang/actions/runs/21241650161 。影响面为所有改动了 `sgl-kernel/` 目录、需要借助评论命令重跑 CI 的 PR。

## 根因
正文披露:`/tag-and-rerun-ci`(及 `/rerun-failed-ci`)在失败时统一使用 `rerun_failed_jobs()`,该 API 只重跑失败 job;而 `sgl-kernel-build-wheels` job 未失败、不会被重跑,artifact 又按原 run 隔离,导致重跑的测试任务在新 run 中找不到可下载的 kernel wheel artifact,报 "Found 0 artifact(s)"。

## 修复方案
在 `scripts/ci/utils/slash_command_handler.py` 中新增 `has_sgl_kernel_changes(pr)`:遍历 `pr.get_files()`,若有文件路径以 `sgl-kernel/` 开头则返回 True(查询异常时打印警告并默认 False,避免不必要整跑)。在 `handle_rerun_failed_ci` 中先调用该函数,重跑失败 workflow 时按结果分流:

```diff
-            # DEBUG
             print(f"Rerunning failed workflow: {run.name} (ID: {run.id})")
             try:
-                # Use rerun_failed_jobs for efficiency on failures
-                run.rerun_failed_jobs()
+                if sgl_kernel_changes:
+                    # Full rerun to ensure sgl-kernel-build-wheels runs
+                    # and produces fresh artifacts for dependent jobs
+                    run.rerun()
+                else:
+                    # Use rerun_failed_jobs for efficiency on failures
+                    run.rerun_failed_jobs()
```

无 sgl-kernel 改动的 PR 仍走高效的 `rerun_failed_jobs()`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 CI 脚本 `scripts/ci/utils/slash_command_handler.py`。影响 `/tag-and-rerun-ci` 与 `/rerun-failed-ci` 的重跑策略:含 sgl-kernel 改动的 PR 将整跑重跑,CI 时长与资源消耗增加(构建 job 也会重跑);`has_sgl_kernel_changes` 在列举 PR 文件异常时默认 False,此时会退回旧行为(重跑后仍可能缺 artifact)。

## 关联
- https://github.com/sgl-project/sglang/actions/runs/21241650161 (正文引用的失败示例)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
