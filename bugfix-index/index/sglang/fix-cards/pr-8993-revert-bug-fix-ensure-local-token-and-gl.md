# PR #8993: Revert "[bug fix] Ensure local token and global token buffers are pointing to different storage "

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8993](https://github.com/sgl-project/sglang/pull/8993) |
| 作者 | ch-wan |
| 合并日期 | 2025-08-09 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回退 #8785:该"确保 local/global buffer 不共享存储"的修复导致精度下降,恢复原来的别名断言。

## 问题现象
正文仅两句:"Reverts sgl-project/sglang#8785" 与 "Accuracy drop"。即被回退的修复(确保 DP attention 的 local token 与 global token buffer 指向不同存储)上线后出现精度下降,故整体撤销。具体下降幅度、复现配置正文未给出。

## 根因
正文披露的回退理由是 "Accuracy drop"——#8785 的改法带来精度下降,但未解释为何下降。(由 diff 推断)机制:#8785 曾在 `_dp_gather_via_all_reduce` 与 `dp_scatter` 中,当检测到 `local_tokens` 与 `global_tokens` 共享同一 `untyped_storage` 时,先 `torch.empty_like` 另建缓冲再执行 `memcpy_triton`(因为两者是 in-place 操作、要求输入输出不别名);本回退删除这一分支,恢复为直接断言 "aliasing between global_tokens and local_tokens not allowed" / "aliasing between local_tokens and global_tokens not allowed"。改法引入的额外缓冲/拷贝路径与精度下降相关。

## 修复方案
改动文件 `python/sglang/srt/layers/dp_attention.py`,恢复两处断言:
- `_dp_gather_via_all_reduce`:删除 `if local_tokens.untyped_storage() is global_tokens.untyped_storage(): global_tokens = torch.empty_like(global_tokens)` 分支,恢复 `assert local_tokens.untyped_storage() is not global_tokens.untyped_storage()`。
- `dp_scatter`:对称地删除 `local_tokens = torch.empty_like(local_tokens)` 分支,恢复对 `local_tokens`/`global_tokens` 的别名断言。

关键 diff(dp_scatter 处):
```diff
-        if local_tokens.untyped_storage() is global_tokens.untyped_storage():
-            # dp_scatter is an in-place operation and requires input and output tensors to not be aliased.
-            # so we create a separate buffer if they share the same storage.
-            local_tokens = torch.empty_like(local_tokens)
+        assert (
+            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
+        ), "aliasing between local_tokens and global_tokens not allowed"
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,即 #8785 合入后至本回退落地之间。
- 回归面:DP attention 的 gather/scatter 数据路径(`dp_gather_via_all_reduce`、`dp_scatter`)。回退后若调用方传入共享存储的 buffer,不再自动另建缓冲,而是直接触发 assert(进程报错)——即从"静默新建缓冲"回到"显式失败";正常传入独立 buffer 的路径不受影响。精度应恢复到 #8785 之前水平。

## 关联
- [sgl-project/sglang#8785](https://github.com/sgl-project/sglang/pull/8785)(正文注明的被 revert 对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
