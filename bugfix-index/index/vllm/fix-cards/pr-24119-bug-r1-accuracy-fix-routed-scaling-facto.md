# PR #24119: [Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24119](https://github.com/vllm-project/vllm/pull/24119) |
| 作者 | yewentao256 |
| 合并日期 | 2025-09-02 |
| labels | deepseek, ready |
| 热度 | 👍 4 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

DeepSeek MoE 内部与外层对 `routed_scaling_factor` 各乘一次(双重相乘)致 R1 精度暴跌;内层改传 1.0 修复。

## 问题现象

DeepSeek-R1 gsm8k 精度下降。用 `lm_eval --model local-completions`(模型 deepseek-ai/DeepSeek-R1,256 并发,5-shot)实测:修复前 flexible-extract 0.6626 / strict-match 0.5201,修复后 0.9530 / 0.9522。修复 issue #24118。

## 根因

正文 Purpose 仅说明该问题由 #23123 引入、修复 issue #24118,未单独写根因分析。「双重相乘」来自标题与 diff 中的注释 "we do scaling outside, set factor to 1.0 to avoid double mul"——即路由缩放因子在外部已经乘过一次,构造 MoE experts 时又传入了 `self.routed_scaling_factor` 被再乘一次(由 diff 推断)。

## 修复方案

修改 `vllm/model_executor/models/deepseek_v2.py`:构造 MoE experts 时把 `routed_scaling_factor=self.routed_scaling_factor` 改为 `1.0`,避免与外层 scaling 重复相乘。

```diff
-            routed_scaling_factor=self.routed_scaling_factor,
+            # we do scaling outside, set factor to 1.0 to avoid double mul
+            routed_scaling_factor=1.0,
```

## 影响与回归风险

- 影响窗口:#23123 合并后至本修复前。
- 回归面:`deepseek_v2.py` 中 MoE experts 的路由缩放因子传参,影响 DeepSeek 系模型(如 R1)的 MoE 输出数值;外部 scaling 逻辑本身未在 diff 中改动(由 diff 推断,scaling 在外层执行)。

## 关联

issue #24118、PR #23123

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
