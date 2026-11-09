# PR #34006: Fix Qwen3-MoE producing garbage with the mori a2a backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34006](https://github.com/sgl-project/sglang/pull/34006) |
| 作者 | yuankaichen-amd |
| 合并日期 | 2026-08-08 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
mori a2a 后端下 Qwen3-MoE 因 is_deepep() 精确匹配未走 MoE 分支,forward_normal 在 mori 已 combine 的输出上再叠加 EP all-reduce 产生乱码;forward 判断补上 is_mori() 修复。

## 问题现象
用 mori a2a 后端跑 Qwen3-MoE 输出 garbage。作者在 8x MI355X(gfx950)驱动完整 Qwen3-30B-A3B RL run 验证:修复后 Megatron 重算与 SGLang rollout 的 log probs 一致到 ~0.005。

## 根因
正文披露:`MoeA2ABackend.is_deepep()` 是精确 enum 匹配,mori 运行会落入 `Qwen3MoeSparseMoeBlock.forward` 的 `forward_normal`;`forward_normal` 在 mori 已经 combine 过的输出上再施加专家并行 all-reduce。deepseek_v2、deepseek_v4、glm4_moe 和 glm4_moe_lite 都已在 `is_deepep()` 旁边 gate 了 `is_mori()`,唯独 qwen3_moe 被漏掉。

## 修复方案
`python/sglang/srt/models/qwen3_moe.py` 的 forward 判断加上 `and not get_moe_a2a_backend().is_mori()`,使 mori 后端不再走 `forward_normal`(一行)。

关键 diff:
```diff
         if (
             not get_moe_a2a_backend().is_deepep()
+            and not get_moe_a2a_backend().is_mori()
             and not get_moe_a2a_backend().is_ascend_fuseep()
         ):
             return self.forward_normal(hidden_states)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 qwen3_moe 的 MoE 前向分支选择;mori 后端下不再走 forward_normal,deepep/ascend_fuseep/标准等其他后端路径不变,无模型数学变化。

## 关联
正文未列出其他 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
