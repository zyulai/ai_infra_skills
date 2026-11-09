# PR #29742: [Bugfix] Fix mismatched nvfp4 gemm output shape

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29742](https://github.com/vllm-project/vllm/pull/29742) |
| 作者 | Isotr0py |
| 合并日期 | 2025-11-30 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
CompressedTensorsW4A4Fp4 的 nvfp4 gemm 恒输出 2d 张量,ViT 等 3d 输入模型形状不匹配;改为保留除末维外的输入形状。

## 问题现象
`CompressedTensorsW4A4Fp4` 量化方案的 gemm 恒定输出 2d hidden states;当模型是 ViT 这类输入为 3d 的场景时,输出形状与输入不匹配导致出错。正文未给出具体报错日志,问题由 issue #29715 报告。

## 根因
正文披露:`CompressedTensorsW4A4Fp4` 的 apply 路径总是输出 2d 的 gemm hidden states(「Currently, `CompressedTensorsW4A4Fp4` always output 2d gemm hidden states」),对输入为 3d 的 ViT 造成形状不匹配。

## 修复方案
改 `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` 的 `apply_weights()`:输出形状不再取 `x.shape[0]`(把 batch 维拍平成 2d),而是保留输入除最后一维外的全部维度,再拼接输出通道数:

```diff
-        output_shape = [x.shape[0], layer.weight_packed.shape[0]]
+        output_shape = [*x.shape[:-1], layer.weight_packed.shape[0]]
```

这样 3d 输入(如 ViT 的 [batch, seq, hidden])得到的输出与输入前缀维度一致。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:所有走 `CompressedTensorsW4A4Fp4`(nvfp4 W4A4)scheme 的 `apply_weights()` 输出;对原本 2d 输入的场景,`[*x.shape[:-1], N]` 与 `[x.shape[0], N]` 等价,行为不变;3d 输入场景由拍平变为保形。

## 关联
- Fix [#29715](https://github.com/vllm-project/vllm/issues/29715)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
