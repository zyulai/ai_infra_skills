# PR #45029: Revert "[Bugfix][CI] Gemma3 Transformers multimodal encoder profiling and build prompt-embedding fixtures"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45029](https://github.com/vllm-project/vllm/pull/45029) |
| 作者 | hmellor |
| 合并日期 | 2026-06-10 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
#44952 对 Gemma3 编码器 token 数的修正是错的,本 PR 回滚其 Transformers 后端部分,恢复按 `num_image_tokens` 计算。

## 问题现象
正文未描述具体现象,只说明被回滚的 #44952「不是该问题的正确修复」(It was not the correct fix for this problem),正确修复是 #39425。

## 根因
正文披露:#44952 中 Transformers 后端部分的改动没有真正解决问题,属于错误修复,正确方案已由 #39425 提供,故回滚。
具体错误机制正文未展开;(由 diff 推断) 被回滚的代码在检测到 Gemma3 processor 时,改用 pre-pooling 的 patch 网格数 `(image_size // patch_size) ** 2 * max(num_image_patches)` 估算 vision encoder 的最大 token 数,覆盖了 HF processor 返回的 `num_image_tokens`,该估算方式不正确。

## 修复方案
回滚 `vllm/model_executor/models/transformers/multimodal.py` 中 #44952 加入的部分:删除 `_get_mm_values`、`_get_max_encoder_tokens` 两个辅助方法及 Gemma3 特判逻辑,`get_max_image_tokens` 恢复为直接返回 processor 结果:

```diff
-        return self._get_max_encoder_tokens(processor, mm_tokens) or image_tokens
-    ...（删除 _get_mm_values / _get_max_encoder_tokens 整段）
+        return image_tokens
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围(根因 PR 为 #44952,合入至本回滚之间的提交受影响)。
- 回归面:仅动 Transformers 后端多模态路径的 `get_max_image_tokens`;Gemma3 及其他多模态模型的最大 image token 计算回到统一的 `num_image_tokens` 逻辑,不再有 Gemma3 分支;原 #44952 声称的 encoder profiling 行为随之消失。

## 关联
- #44952(被回滚的 PR)
- #39425(正文指认的正确修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
