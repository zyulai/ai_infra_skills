# PR #28972: [Bugfix] Fix precision loss in LoRA-wrapped RowParallelLinear by fusing bias into GEMM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28972](https://github.com/vllm-project/vllm/pull/28972) |
| 作者 | prashanth058 |
| 合并日期 | 2025-11-20 |
| labels | ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

LoRA 包装的 RowParallelLinear 把 bias 单独用 bfloat16 相加,与未包装层的 GEMM 内融合 bias 相比产生精度损失;改为仅在 rank 0 把 bias 传给 apply() 融合进 GEMM 修复。

## 问题现象

LoRA 包装的 RowParallelLinear 与合并权重后的层相比,输出存在精度差异——即使 LoRA 权重全为零,两条路径的结果也不一致。触发条件是使用 LoRA 包装该层时:包装层把 bias 作为一次独立的 bfloat16 加法,而未包装层则把 bias 融合进 GEMM kernel,导致数值路径不同、结果不一致。

## 根因

正文披露:LoRA 包装的 RowParallelLinear 之前把 bias 作为单独的 bfloat16 操作相加,而不是像未包装层那样把 bias 融合进 GEMM kernel。融合 kernel 可以在更高精度(FP32)累加后再转成 bfloat16,而分开相加会引入额外的舍入误差,因此即便 LoRA 权重为零,LoRA 包装与合并权重的结果也存在偏差。

## 修复方案

修改 `vllm/lora/layers/row_parallel_linear.py`:forward 中先按 TP rank 决定是否传 bias——只在 rank 0 且 `skip_bias_add=False` 时传 `self.base_layer.bias`,否则传 None——然后直接 `self.apply(input_parallel, bias_)`,让量化方法把 bias 加法融合进 GEMM kernel;`apply` 则把 bias 透传给 `self.base_layer.quant_method.apply`。关键 diff:

```diff
-        output_parallel = self.apply(input_parallel)
+        bias_ = (
+            None
+            if (self.tp_rank > 0 or self.base_layer.skip_bias_add)
+            else self.base_layer.bias
+        )
+        output_parallel = self.apply(input_parallel, bias_)
```

同时删除了原来 all-reduce 之后单独 `output_ + self.base_layer.bias` 的加法分支,`output_bias` 改为仅在 `skip_bias_add` 时返回 bias。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:改动仅针对 `vllm/lora/layers/row_parallel_linear.py` 的 forward/apply 逻辑。使用 LoRA 包装 RowParallelLinear 且 TP>1 的场景下,bias 现在在 all-reduce 之前于 rank 0 加入再规约,数值路径与未包装层一致;需要注意 skip_bias_add/return_bias 分支的行为对齐是否在所有量化方法下成立。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
