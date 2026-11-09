# PR #28285: [fix] Revert "fixing mm placeholder replacement issue with gemma3"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28285](https://github.com/vllm-project/vllm/pull/28285) |
| 作者 | khluu |
| 合并日期 | 2025-11-07 |
| labels | ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
回滚 #27538:gemma3 多模态占位符替换中换行 token 顺序的改动破坏 Multi-Modal Test Extended 3。

## 问题现象
Multi-Modal Test Extended 3 失败——正文仅有一句话:"Revert https://github.com/vllm-project/vllm/pull/27538 since it's breaking Multi-Modal Test Extended 3"。未提供具体报错或失败详情。

## 根因
正文披露:PR #27538("fixing mm placeholder replacement issue with gemma3")破坏了 Multi-Modal Test Extended 3,因此整体回滚。(由 diff 推断)#27538 把 `_find_mm_placeholders` 中 `newline_3` 的替换 token 顺序从 `[newline_1, newline_2]` 改成了 `[newline_2, newline_1]`,该顺序变化导致测试失败。

## 修复方案
直接回滚 #27538 在 `vllm/model_executor/models/gemma3_mm.py` 的改动,恢复 `get_repl_toks` 中 `newline_3` 的替换顺序:

```diff
         def get_repl_toks(tok: int) -> list[int]:
             if tok == newline_3:
-                return [newline_2, newline_1]
+                return [newline_1, newline_2]
             if tok == newline_4:
                 return [newline_2, newline_2]
```

## 影响与回归风险
- 影响窗口:#27538 合并之后至本回滚之前,gemma3 多模态占位符替换走被改动的顺序,Multi-Modal Test Extended 3 失败。
- 回归面:`gemma3_mm.py` `_find_mm_placeholders` 的占位符替换路径。回滚后恢复原顺序、测试恢复;注意 #27538 原本要修复的"mm placeholder replacement issue with gemma3"会随之回归——正文未说明是否有替代修复方案。

## 关联
- PR #27538(被回滚的改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
