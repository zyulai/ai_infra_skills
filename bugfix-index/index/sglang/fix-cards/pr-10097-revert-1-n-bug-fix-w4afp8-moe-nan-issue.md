# PR #10097: Revert "[1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel) (#9953)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10097](https://github.com/sgl-project/sglang/pull/10097) |
| 作者 | zhyncs |
| 合并日期 | 2025-09-06 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
回退 #9953 对 w4a8 grouped GEMM 输出类型的改动,把 ElementC/ElementD 从 bf16 恢复为 FP16 默认输出类型。

## 问题现象
正文未描述具体现象。PR 为 revert 声明:「This reverts commit f78b7fd16dbfe32c2ee73c1f3fef49fc1257b27f」(即 #9953「[1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel)」),Motivation/Modifications/测试各节均为空模板,未附任何复现或验证数据。

## 根因
正文未披露根因。从 diff 描述回退所暗示的机制(由 diff 推断):#9953 曾把 `w4a8_grouped_mm_c3x.cuh` 的 CUTLASS grouped GEMM 输出类型 `ElementC`/`ElementD` 从 `cutlass::half_t`(FP16 默认输出类型)改成 `cutlass::bfloat16_t`;本 PR 将其恢复原状,以撤销该 NaN 修复带来的影响。

## 修复方案
revert 单文件 `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` 的类型定义:

```python
-using ElementC = cutlass::bfloat16_t;      // Output type
-using ElementD = ElementC;                 // Output type
+using ElementC = cutlass::half_t;          // Default output type (FP16)
+using ElementD = ElementC;                 // Default output type (FP16)
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:sgl-kernel w4a8 CUTLASS grouped GEMM 的输出 dtype;#9953 原本针对的 w4afp8 MoE NaN 场景会随 revert 一并回到修复前状态,正文未提供后续处理计划。

## 关联
- #9953(被回退的 PR,标题中引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
