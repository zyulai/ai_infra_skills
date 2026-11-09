# PR #9953: [1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9953](https://github.com/sgl-project/sglang/pull/9953) |
| 作者 | yuhyao |
| 合并日期 | 2025-09-03 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

w4afp8 grouped GEMM kernel 输出类型由 FP16 改为 BF16(与 scale 类型一致),修复 w4afp8 MoE NaN 问题。

## 问题现象

w4afp8 MoE 出现 NaN 问题(标题所述)。正文未描述具体现象、复现步骤或日志,精度与性能测试均指向配套 PR #9918 查看。

## 根因

正文未披露 NaN 产生的机理分析,Modifications 仅说明改动内容:「Update the w4afp8 grouped gemm kernel to use bfloat16 as the output type」。(由 diff 推断)该 CUTLASS w4a8 grouped GEMM 模板中 scale 类型本为 `cutlass::bfloat16_t`,而输出类型 `ElementC`/`ElementD` 为 `cutlass::half_t`(FP16),输出与 scale 类型不一致;标题表明该 FP16 输出类型与 NaN 有关,改为 BF16 后消除。

## 修复方案

改 `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` 的输出类型定义:

```diff
-using ElementC = cutlass::half_t;          // Default output type (FP16)
-using ElementD = ElementC;                 // Default output type (FP16)
+using ElementC = cutlass::bfloat16_t;      // Output type
+using ElementD = ElementC;                 // Output type
```

即 grouped GEMM 的 C/D 输出类型从 FP16 改为 BF16;accumulator(float)与 scale(bfloat16_t)不变。

## 影响与回归风险

- 回归面:sgl-kernel 中 w4a8/w4afp8 CUTLASS grouped GEMM 的输出 dtype——下游消费该 kernel 输出的 w4afp8 MoE 前向路径需按 BF16 输出对接。注意:作者更新说明本 PR 缺少 unit test,更新版本见 PR #10108。

## 关联

- [PR #9918](https://github.com/sgl-project/sglang/pull/9918)(sglang 侧配套改动,精度/性能数据所在)
- [PR #10108](https://github.com/sgl-project/sglang/pull/10108)(带 unit test 的更新版本)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
