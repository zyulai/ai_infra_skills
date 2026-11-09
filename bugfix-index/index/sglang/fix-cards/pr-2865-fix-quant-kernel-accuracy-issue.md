# PR #2865: Fix quant kernel accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2865](https://github.com/sgl-project/sglang/pull/2865) |
| 作者 | ispobock |
| 合并日期 | 2025-01-13 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

int8 per-token 量化 kernel 用乘法 `x * (127/absmax)` 替代除法 `x / scale_x` 后 round,评估精度更佳。

## 问题现象

正文未描述具体现象,仅在 Motivation 中说明:改用乘法代替除法「似乎能在评估中获得更好的精度」(Use multiplication instead of division seems can get better accuracy on evaluation),未附具体评估数字、日志或复现步骤。

## 根因

正文未披露根因,只给出动机(乘法比除法精度更好)。(由 diff 推断)`_per_token_quant_int8` 的 Triton kernel 原先计算 `scale_x = absmax / 127` 后再算 `x / scale_x` 并 round;除法引入的舍入误差被改为等价的乘法形式 `x * (127 / absmax)`,即先算出 127/absmax 的乘子再相乘,以换取更准确的量化结果。

## 修复方案

改 `python/sglang/srt/layers/quantization/int8_kernel.py` 的 `_per_token_quant_int8`:

```diff
-    x_q = tl.extra.cuda.libdevice.round(x / scale_x).to(tl.int8)
+    x_q = x * (127 / absmax)
+    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)
```

量化 scale 的计算(`scale_x = absmax / 127`)与存储保持不变,仅反算 quantized 值的方式由除法改为乘法。

## 影响与回归风险

- 回归面:所有走 `_per_token_quant_int8` 的 int8 per-token 量化路径的数值结果(极小差异级别);kernel 输出的取值含义不变,`scale_x` 仍照旧写出。正文未附评估基准与性能数据,精度收益幅度未知。

## 关联

- cc @grimoire(正文提及,无 issue/PR 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
