# PR #5777: Revert "Revert "fix: import vllm_rotary_embedding error when head_size not in 64, 128, 256, 512""

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5777](https://github.com/sgl-project/sglang/pull/5777) |
| 作者 | merrymercy |
| 合并日期 | 2025-04-27 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
撤销 #5772 的回退、重新落地原修复:按条件延迟导入 vllm rotary_embedding,解决 head_size 不在支持列表时的导入错误。

## 问题现象
正文仅一句 "Reverts sgl-project/sglang#5772",未描述具体现象。从标题可知症状为:head_size 不在 64、128、256、512 之中时,出现 vllm_rotary_embedding 相关导入错误。影响面、触发日志正文均未给出。

## 根因
正文未披露根因,只说明本 PR 撤销 #5772(即重新引入被回退的原修复)。(由 diff 推断)机制:原实现把 `from vllm._custom_ops import rotary_embedding as vllm_rotary_embedding` 放在模块顶层、且仅位于 `_is_cuda` 为假的 `else` 分支;修复改为在 `__init__` 中按 `not _is_cuda or self.head_size not in [64, 128, 256, 512]` 条件延迟导入并绑定到 `self.vllm_rotary_embedding`。这暗示 bug 机制是:CUDA 环境下 head_size 不在支持列表时 forward 仍需 vllm 的 rotary_embedding 实现,而模块级导入分支不会在该场景执行,导致引用出错。

## 修复方案
通过 revert #5772 恢复原修复的改动,作用于 `python/sglang/srt/layers/rotary_embedding.py`:
1. 删除模块顶层 `else` 分支的无条件导入;
2. 在 `__init__` 中当 `not _is_cuda or head_size not in [64, 128, 256, 512]` 时才 `from vllm._custom_ops import rotary_embedding`,并赋给 `self.vllm_rotary_embedding`;
3. `forward_cuda` 中调用点由模块级 `vllm_rotary_embedding(...)` 改为 `self.vllm_rotary_embedding(...)`。

关键 diff:
```diff
-else:
-    from vllm._custom_ops import rotary_embedding as vllm_rotary_embedding
+        if not _is_cuda or self.head_size not in [64, 128, 256, 512]:
+            from vllm._custom_ops import rotary_embedding
+
+            self.vllm_rotary_embedding = rotary_embedding
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,仅涉及 #5772 回退与本次重新落地之间。
- 回归面:改动集中在 rotary_embedding.py 的导入时机与 forward 调用点。非 CUDA 平台与 head_size 不在 {64, 128, 256, 512} 的模型会走新导入分支;CUDA 上标准 head_size 的路径不再执行模块级 vllm 导入。导入失败或 vllm 版本不兼容的场景将影响这些路径的可用性。

## 关联
- [sgl-project/sglang#5772](https://github.com/sgl-project/sglang/pull/5772)(正文注明的被 revert 对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
