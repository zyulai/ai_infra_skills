# PR #8766: Fix mismatch between padded_scales shape and reshape dimensions in modelopt quantization

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8766](https://github.com/sgl-project/sglang/pull/8766) |
| 作者 | ovowei |
| 合并日期 | 2025-08-12 |
| labels | ready-to-merge |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
modelopt 量化中 padded_scales 形状为 (B, M_padded, K_padded) 却被 reshape 成未 padded 的 (B, M, K)/(M, K),改回 padded 维度。

## 问题现象
在 SGLang 中使用 ModelOpt 做 NVFP4 量化推理时,`padded_scales` 的形状与后续 reshape 维度不匹配,可能导致运行时报错或错误的内存对齐(正文原话:"potential runtime errors or incorrect memory alignment"),尤其出现在 NVFP4 quantized inference 场景。正文未给出具体报错信息或复现命令。

## 根因
正文(Modifications 一节)明确披露:当前实现中 `padded_scales` 以 `(B, M_padded, K_padded)` 形状初始化,但随后按 `scale_ndim` 被 reshape 成 `(B, M, K)` 或 `(M, K)`——即用未做 padding 的目标维度去 reshape 含 padding 的数据,两者不匹配,导致潜在的运行时错误或内存对齐问题。swizzle 路径(`swizzle_blockscale`)存在同样的 reshape 维度错误。

## 修复方案
改 `python/sglang/srt/layers/quantization/modelopt_quant.py` 中两处 reshape,把目标维度从 padded 前的 `M, K` 改为 padded 后的 `M_padded, K_padded`:

- `process_weights_after_loading`(权重加载后处理):`scale_ndim == 2` 时 `padded_scales.reshape(M_padded, K_padded)`,否则 `reshape(B, M_padded, K_padded)`;
- `swizzle_blockscale`(scale 重排):同样改为 `reshape(M_padded, K_padded)` / `reshape(B, M_padded, K_padded)`。

```diff
         padded_scales = (
-            padded_scales.reshape(M, K)
+            padded_scales.reshape(M_padded, K_padded)
             if scale_ndim == 2
-            else padded_scales.reshape(B, M, K)
+            else padded_scales.reshape(B, M_padded, K_padded)
         )
```

`layer.weight_scale_interleaved` 由此持有与真实 padded 布局一致的形状。

## 影响与回归风险
- 回归面:仅影响 modelopt 量化路径中 scale 张量的输出形状——`weight_scale_interleaved` 与 `swizzle_blockscale` 返回值从 `(M, K)`/`(B, M, K)` 变为 `(M_padded, K_padded)`/`(B, M_padded, K_padded)`;消费这些 scale 的 NVFP4 kernel 本就按 padded 布局读取,属于形状契约修正;非 modelopt 量化路径不受影响。

## 关联
正文未列出
