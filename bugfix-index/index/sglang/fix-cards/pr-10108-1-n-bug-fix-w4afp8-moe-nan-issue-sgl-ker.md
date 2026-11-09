# PR #10108: [1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel, fixed)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10108](https://github.com/sgl-project/sglang/pull/10108) |
| 作者 | yuhyao |
| 合并日期 | 2025-09-08 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
w4afp8 MoE grouped GEMM 输出类型误用 FP16 导致 NaN,改为 bfloat16 修复。

## 问题现象
标题指出 w4afp8 MoE 出现 NaN 问题;正文未描述具体现象与触发条件,精度与性能验证结果均指向 [PR#9918](https://github.com/sgl-project/sglang/pull/9918)。本 PR 是该修复的 sgl-kernel 侧改动(1/N 分批提交)。

## 根因
正文未披露根因分析,仅说明"将 w4afp8 grouped gemm 内核的输出类型更新为 bfloat16"。

从 diff 看,`sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` 中 CUTLASS grouped GEMM 的输出类型 `ElementC`/`ElementD` 原为 `cutlass::half_t`(FP16),改动后统一为 `cutlass::bfloat16_t`;即输出 dtype 与模型实际使用的 bfloat16 不一致(由 diff 推断),对应标题所述的 NaN 现象。

## 修复方案
1. `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh`:输出类型改为 bfloat16:

```diff
-using ElementC = cutlass::half_t;          // Default output type (FP16)
-using ElementD = ElementC;                 // Default output type (FP16)
+using ElementC = cutlass::bfloat16_t;      // Output type
+using ElementD = ElementC;                 // Output type
```

2. `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py`:单专家/多专家两组测试的输出张量 dtype 从 `torch.float16` 改为 `torch.bfloat16`;参考实现 `ref_grouped_gemm` 的矩阵乘改为在 float32 下计算后再转目标 dtype,与内核行为对齐。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:w4a8/int4-FP8 CUTLASS grouped GEMM 内核(`cutlass_w4a8_moe_mm`)的所有调用路径,输出 dtype 变为 bfloat16,依赖 FP16 输出的上游代码需同步;配套测试 `test_cutlass_w4a8_moe_mm.py` 同步更新。

## 关联
- [PR#9918](https://github.com/sgl-project/sglang/pull/9918)(w4afp8 主 PR,精度/基准结果)
- [PR#9953](https://github.com/sgl-project/sglang/pull/9953)(本 PR 声明修复该 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
