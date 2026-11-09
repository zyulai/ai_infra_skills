# PR #10144: [MoE] fix: incorrect weight initialization for  cutlass_fused_experts_fp8

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10144](https://github.com/sgl-project/sglang/pull/10144) |
| 作者 | ch-wan |
| 合并日期 | 2025-09-08 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

cutlass_fused_experts_fp8 的 c_strides2 用错误尺寸初始化权重元数据,改用 hidden_size 修复。

## 问题现象

正文未描述具体现象。Motivation 一节仅写「fix #10138」,未给出复现步骤、日志或影响面描述。

## 根因

正文未披露根因(Modifications 一节为空,仅有模板占位注释)。

从 diff 描述改动所暗示的机制(由 diff 推断):`create_weights` 中构造 `c_strides2` 这个 per-expert 的 int64 张量时,填充值错误地使用了 `intermediate_size_per_partition`,而该 stride 语义上应基于 `hidden_size`;尺寸来源错误导致 cutlass_fused_experts_fp8 路径的权重初始化元数据不正确。

## 修复方案

改动集中在 `python/sglang/srt/layers/quantization/fp8.py` 的 `create_weights`:将 `c_strides2 = torch.full((num_experts,), ...)` 的填充值由 `intermediate_size_per_partition` 改为 `hidden_size`,其余参数(device、dtype=torch.int64)不变。

关键 diff 摘录:

```diff
                 self.c_strides2 = torch.full(
                     (num_experts,),
-                    intermediate_size_per_partition,
+                    hidden_size,
                     device=w2_weight.device,
                     dtype=torch.int64,
                 )
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅影响 FP8 量化下 cutlass_fused_experts_fp8 权重创建时 `c_strides2` 的取值,即该 kernel 的 stride 元信息;单行改动,不涉及其他量化路径或 forward 逻辑。正文未提供 accuracy 测试结果(Checklist 均未勾选)。

## 关联

- #10138(正文 Motivation 写明 fix #10138)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
