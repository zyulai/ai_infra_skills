# PR #46860: [Bugfix][Quantization] Fix W8A8 int-quantized scheme selection regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46860](https://github.com/vllm-project/vllm/pull/46860) |
| 作者 | HDCharles |
| 合并日期 | 2026-06-29 |
| labels | bug, quantization, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#46389 误把 W8A8O8 判定 and 写成 or,致 W8A8 int8 静态量化走错 scheme,改回并补 13 例测试。

## 问题现象
PR #46389 合入后,W8A8 int8_static 的 compressed-tensors 模型 scheme 选择被破坏。触发条件:权重 int8 量化、输入为 static int8 但输出不是 static int8 的配置——正文称 "static_int8_in without static_int8_out would go into the wrong pathway and cause issues"。作者给出验证方式 `vllm serve nm-testing/w8a8_static_asym-e2e`(修复后 working)。

## 根因
正文明确披露:#46389 在改 W8A8O8 条件时意外把

`(is_static_int8_in and is_static_int8_out)` 改成了 `(is_static_int8_in or is_static_int8_out)`

。于是「只有输入静态 int8、输出不静态」的 W8A8 配置也满足 `_is_wNa8o8_int`,被误分到 WNA8O8Int(fake-quant)路径,而不是真正 int8 gemm 的 W8A8Int8 路径。

## 修复方案
`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` 的 `_is_wNa8o8_int` 把条件改回 `and`:

```diff
-        return is_intN_weight and (is_static_int8_in or is_static_int8_out)
+        return is_intN_weight and (is_static_int8_in and is_static_int8_out)
```

同时在 `tests/quantization/test_compressed_tensors.py` 新增参数化测试 `test_scheme_selection`,13 个用例覆盖 W8A8 int-quantized(channel/tensor 权重、sym/asym/dynamic 输入)、W8A8O8、W4A8O8、以及 W2-W8 weight-only pack-quantized(含新的 3/5/6/7-bit)。正文验证:修复后全部通过,不带修复时其中 2 个用例失败,能正确捕获该回归。

## 影响与回归风险
- 影响窗口:#46389 合入到本修复之间的版本,期间 W8A8 int8_static(仅输入静态)模型会选错 scheme。
- 回归面:`_is_wNa8o8_int` 只服务 compressed-tensors int-quantized 的 scheme 分派;改回 `and` 后 WNA8O8Int 仅在输入与输出都为静态 int8 时才匹配,其余组合回落到 W8A8Int8 等原有路径。对输出量化本来就不存在的纯 W8A8 用户是行为修复;对同时声明输入输出静态量化的 W8A8O8/W4A8O8 用户判定不变。

## 关联
- vllm-project/vllm#46389(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
