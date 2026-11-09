# PR #15932: feat: bugfix and accuracy fix for stablelm2_1_6b

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15932](https://github.com/sgl-project/sglang/pull/15932) |
| 作者 | McZyWu |
| 合并日期 | 2025-12-28 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

stablelm-2-1_6b 在 NPU 上准确率为 0.0;让 NPU 的 rotary embedding 在 torch compile 开启时也走 native 实现,准确率恢复到 0.2。

## 问题现象

stablelm-2-1_6b 模型在 NPU 上推理的准确率为 0.0(正文原话:"accuracy fix for stablelm-2-1-6b for npu, previously it was 0.0")。触发条件:NPU 上运行该模型。修复后准确率为 0.2(正文 Accuracy Tests 一栏原文仅写"0.2",未给出评测集与详细数据)。

## 根因

正文未披露根因,只说明这是针对 NPU 上 stablelm-2-1_6b 的 accuracy fix。(由 diff 推断)`rotary_embedding.py` 的 `forward_npu` 原本仅在 `query` 为 bfloat16 且 `cos_sin_cache` 为 float 时回退 `forward_native`,其余情况走 NPU 专用实现;当开启 `SGLANG_ENABLE_TORCH_COMPILE` 时 NPU 实现路径的 rotary embedding 结果不正确,导致整体输出无效、准确率归零。

## 修复方案

改 `python/sglang/srt/layers/rotary_embedding.py` 的 `forward_npu`:在原有 bfloat16/float 回退条件之外,增加 `SGLANG_ENABLE_TORCH_COMPILE` 环境变量作为回退条件,使 torch compile 开启时也走 `forward_native`:

```diff
-        if query.dtype == torch.bfloat16 and self.cos_sin_cache.dtype == torch.float:
+        if (query.dtype == torch.bfloat16 and self.cos_sin_cache.dtype == torch.float) or get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE"):
             return self.forward_native(positions, query, key, offsets)
```

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:NPU 上 `forward_npu` 的 rotary embedding 路径,仅当 `SGLANG_ENABLE_TORCH_COMPILE` 开启时行为改变(回退 native 实现);未开启该环境变量的 NPU 推理路径不受影响,bfloat16 + float cache 的既有回退条件保持不变。未开启 compile 的其他环境完全不受影响。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
