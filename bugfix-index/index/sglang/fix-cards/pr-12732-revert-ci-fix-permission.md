# PR #12732: Revert "[ci] fix permission"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12732](https://github.com/sgl-project/sglang/pull/12732) |
| 作者 | key4ng |
| 合并日期 | 2025-11-06 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #12729:labeler workflow 的权限改动引发问题,恢复 `contents: read` + `pull-requests: write`。

## 问题现象
正文未描述具体现象,仅一句话说明本 PR 回滚 #12729([ci] fix permission)。PR 打了 run-ci 标签,合并日期 2025-11-06,正文没有给出失败的工作流、报错日志或复现步骤。

## 根因
正文未披露根因。从 diff 看,#12729 曾把 `.github/workflows/labeler.yml` 顶层的 workflow 权限从 `contents: read` + `pull-requests: write` 改为仅 `pull-requests: read`;本 PR 将其恢复(由 diff 推断)。该文件是 PR labeler 工作流(`jobs: label`,监听 opened/synchronize/reopened),把 `pull-requests` 权限降到 read 后,工作流对 PR 的写操作(打标签)可能因权限不足而失败(由 diff 推断)。

## 修复方案
恢复 `.github/workflows/labeler.yml` 的 permissions 块:

```diff
 permissions:
-  pull-requests: read
+  contents: read
+  pull-requests: write
```

即回到 #12729 之前的权限配置:读取仓库内容 + 对 pull-requests 可写。

## 影响与回归风险
- 影响窗口:正文未说明(#12729 合并至本回滚之间)。
- 回归面:仅 CI 基础设施——labeler 工作流的 GITHUB_TOKEN 权限,不触及任何运行时代码或模型路径;labeler 恢复对 PR 的写权限,行为回到 #12729 之前。

## 关联
- [#12729](https://github.com/sgl-project/sglang/pull/12729)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
