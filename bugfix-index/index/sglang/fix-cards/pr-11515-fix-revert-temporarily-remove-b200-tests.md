# PR #11515: fix: revert temporarily remove b200 tests

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11515](https://github.com/sgl-project/sglang/pull/11515) |
| 作者 | zhyncs |
| 合并日期 | 2025-10-12 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
恢复 CI 中被临时注释掉的 4-GPU B200 测试 job,重新在 Blackwell B200 runner 上跑 per-commit 测试套件。

## 问题现象
正文 Motivation/Modifications 等章节均为空模板,未描述具体现象。从标题可知:此前 B200 测试被"临时移除"(temporarily remove b200 tests),本 PR 将其恢复(revert)。症状层面即 B200 GPU 的 CI 测试不再随 PR 运行;具体移除原因正文未给出。

## 根因
正文未披露根因。(由 diff 推断)机制:diff 全部位于 `.github/workflows/pr-test.yml`,把此前被逐行注释掉的 `unit-test-backend-4-gpu-b200` job 恢复为有效定义——即撤销"临时移除"操作。恢复后的 job 依赖 `check-changes`、`unit-test-backend-2-gpu`、`sgl-kernel-build-wheels`,在 `4-gpu-b200` runner 上安装依赖(`IS_BLACKWELL=1`,CUDA 12.9 wheel)并执行 `python3 run_suite.py --suite per-commit-4-gpu-b200 --auto-partition-id 0 --auto-partition-size 1 --timeout-per-file 3600`。

## 修复方案
改动文件 `.github/workflows/pr-test.yml`:取消 `unit-test-backend-4-gpu-b200` job 的全部注释,恢复其完整定义,位置保持在 `per-commit-8-gpu-deepep` job 之后、`pr-test-finish` 之前。

关键 diff(恢复后的 job 头部):
```diff
-  #unit-test-backend-4-gpu-b200:
-  #  needs: [check-changes, unit-test-backend-2-gpu, sgl-kernel-build-wheels]
-  #  if: always() && !failure() && !cancelled() &&
-  #      ((needs.check-changes.outputs.main_package == 'true') || (needs.check-changes.outputs.sgl_kernel == 'true'))
-  #  runs-on: 4-gpu-b200
+  unit-test-backend-4-gpu-b200:
+    needs: [check-changes, unit-test-backend-2-gpu, sgl-kernel-build-wheels]
+    if: always() && !failure() && !cancelled() &&
+        ((needs.check-changes.outputs.main_package == 'true') || (needs.check-changes.outputs.sgl_kernel == 'true'))
+    runs-on: 4-gpu-b200
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间。
- 回归面:纯 CI 配置改动,不影响任何运行时代码。影响 PR 测试流水线:满足 `main_package` 或 `sgl_kernel` 变更条件时,B200 4-GPU 测试将重新执行,可能占用 `4-gpu-b200` runner 资源并延长 CI 时长;若 B200 环境仍有未修复问题,该 job 的失败会重新出现在 PR 检查中。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
