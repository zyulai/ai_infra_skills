# PR #24208: [CI] Fix stage-b-test-4-gpu-b200 silently skipped, hanging wait-for-stage-b

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24208](https://github.com/sgl-project/sglang/pull/24208) |
| 作者 | Fridge003 |
| 合并日期 | 2026-05-01 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

CI job 的 runs-on 引用已删除的 workflow 输出被静默跳过;恢复为 b200_runner 使 wait-for-stage-b 不再空等。

## 问题现象

CI 中 `stage-b-test-4-gpu-b200` job 被 GitHub Actions 静默跳过、从不运行;与此同时 `wait-for-stage-b` 仍按 `expected_count: 1`(总计 27)等待该 job,永远轮询直到超时。正文给出实际观测:[run 25195250632 的 job 73890630667](https://github.com/sgl-project/sglang/actions/runs/25195250632/job/73890630667) 每轮都输出 `Progress: 26/26 jobs completed (expected 27)`。

## 根因

正文披露:`stage-b-test-4-gpu-b200` 的 `runs-on:` 引用 `needs.check-changes.outputs.b200_low_disk_runner`,而该输出在 #24163 的 revert 中被移除,#23505 引入的消费端改动却没有一起回退;GitHub Actions 对 `runs-on` 求值为空的 job 会静默跳过,job 从未执行,`wait-for-stage-b` 等不到它而无限轮询。

## 修复方案

改 `.github/workflows/pr-test.yml`:把该 job 的 `runs-on` 恢复为 #23505 之前的值 `b200_runner`,使消费端与 `check-changes` 实际产出的输出重新对齐;同时删除关于 `*-low-disk` 标签池的三行注释。

```diff
-    # The `*-low-disk` label (resolved by `set-runner` to `4-gpu-b200-low-disk` or
-    # `4-gpu-b200-kernel-low-disk`) is advertised by both the existing large-disk B200
-    # runners and the new low-disk runner, so this job can land on either pool.
-    runs-on: ${{ needs.check-changes.outputs.b200_low_disk_runner }}
+    runs-on: ${{ needs.check-changes.outputs.b200_runner }}
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 CI workflow 中该 job 的 runner 选择;无生产代码改动。风险点为该 job 需要确实存在 `b200_runner` 输出对应的 runner 池(正文测试计划要求确认 job 落在 4-gpu-b200 或 4-gpu-b200-kernel runner 上,且 wait-for-stage-b 不再 27/27 超时)。

## 关联

- https://github.com/sgl-project/sglang/pull/24163
- https://github.com/sgl-project/sglang/pull/23505
- https://github.com/sgl-project/sglang/actions/runs/25195250632/job/73890630667

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
