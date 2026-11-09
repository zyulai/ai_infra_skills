# PR #22723: [Fix] Fix accuracy bug in Flashmla sparse MLA kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22723](https://github.com/sgl-project/sglang/pull/22723) |
| 作者 | Fridge003 |
| 合并日期 | 2026-04-15 |
| labels | run-ci, sgl-kernel |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
FlashMLA sparse MLA 内核精度错误;将 sgl-kernel 依赖的 FlashMLA 固定版本升级到修复后 commit 解决。

## 问题现象
正文未描述具体现象。PR 描述的 Motivation 部分仅一句「Close #21291」,即该 PR 用于关闭 issue #21291,现象细节需参见该 issue;本 pack 未包含 issue 正文,无法从素材中复述症状与触发条件。

## 根因
正文未披露根因(Modifications 一节为空)。(由 diff 推断)唯一改动是 `sgl-kernel/cmake/flashmla.cmake` 中 FlashMLA 的 `GIT_TAG` 由 `9804b12079e4c873514d3457aa588d3ccf40da28` 换成 `abb54777d4e08c8054c238f59889b52d4e9f0896`,即精度 bug 位于上游 sgl-project/FlashMLA 仓库的 sparse MLA kernel 实现中,sglang 侧通过重新固定(commit bump)引入上游修复,未改动 sglang 自身代码。

## 修复方案
仅一处改动:`sgl-kernel/cmake/flashmla.cmake` 的 FetchContent 声明中,FlashMLA 源码 `GIT_TAG` 由旧 commit `9804b12079e4c873514d3457aa588d3ccf40da28` 更新为 `abb54777d4e08c8054c238f59889b52d4e9f0896`,使 sgl-kernel 构建时拉取包含修复的 FlashMLA 版本。

```diff
 FetchContent_Declare(
     repo-flashmla
     GIT_REPOSITORY https://github.com/sgl-project/FlashMLA
-    GIT_TAG 9804b12079e4c873514d3457aa588d3ccf40da28
+    GIT_TAG abb54777d4e08c8054c238f59889b52d4e9f0896
     GIT_SHALLOW OFF
 )
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:所有使用 FlashMLA sparse MLA kernel 的路径(需重新编译 sgl-kernel 才会拉到新 commit);FlashMLA 上游两个 commit 之间的其他行为变化(修复之外的性能/接口变化)会一并带入,正文与 diff 均未列出上游 commit 内容,具体回归面无法进一步确认。

## 关联
- Issue #21291:https://github.com/sgl-project/sglang/issues/21291
- 其余 issue/PR:正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
