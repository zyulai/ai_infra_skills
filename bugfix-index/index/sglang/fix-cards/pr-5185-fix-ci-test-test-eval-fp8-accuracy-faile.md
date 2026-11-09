# PR #5185: Fix ci test "test_eval_fp8_accuracy" failed

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5185](https://github.com/sgl-project/sglang/pull/5185) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-04-09 |
| labels | high priority |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

AMD CI 的 FP8 精度评测失败;当权重为 per-tensor 缩放时强制激活也用 per-tensor 量化,解除 CI 阻塞。

## 问题现象

AMD CI 中的 `test_eval_fp8_accuracy` 测试失败,阻塞了 AMD CI 流程,导致一些 PR 无法合并。Motivation 原文:"Unblock the AMD CI error procedure, to allow some PRs to be merged"。正文未描述具体报错日志或失败数值。

## 根因

正文未系统披露根因,只给出修复思路:当权重 scale 也是 per-tensor 量化时,激活的 input_scale 应强制使用 per-tensor 量化(Modifications 原文:"force the input_scale use per-tensor quant when weight scale is also per-tenor quant")。diff 中作者留下的 TODO 注释补充了机制背景:当前分支在权重 per-tensor(`weight_scale.numel() == 1`)而激活 per-token(`x_scale.numel() > 1`)的组合下存在 torch.compile 错误(见该文件下方 line#308 一带),彻底方案需要后续支持 per-tensor 激活缩放并解决该编译错误(该说明来自 diff 注释,由 diff 推断)。

## 修复方案

改 `python/sglang/srt/layers/quantization/fp8_utils.py` 的 `apply_fp8_linear` 非 CUDA 分支:在 `_is_hip` 且 `weight_scale.numel() == 1`(权重为 per-tensor 量化)时,改用 `per_token_group_quant_fp8(input_2d, group_size=input_2d.shape[1])` —— group size 等于整行,等效 per-tensor 激活量化:

```diff
-            qinput, x_scale = ops.scaled_fp8_quant(
-                input_2d, input_scale, use_per_token_if_dynamic=use_per_token_if_dynamic
-            )
+            if _is_hip and weight_scale.numel() == 1:
+                qinput, x_scale = ops.scaled_fp8_quant(
+                    input_2d,
+                    input_scale,
+                    use_per_token_if_dynamic=use_per_token_if_dynamic,
+                )
+            else:
+                qinput, x_scale = per_token_group_quant_fp8(
+                    input_2d, group_size=input_2d.shape[1]
+                )
```

作者标注为临时方案(TODO),最终方案是支持 per-tensor 激活缩放并解决上述 torch.compile 错误。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;影响非 CUDA(HIP/ROCm)FP8 动态量化线性层的激活量化路径。
- 回归面:仅 `_is_hip` 且权重 per-tensor 的分支改变行为(激活从 per-token 改为等效 per-tensor 量化),该组合下精度特性会有变化;CUDA 路径与权重 per-token 的 HIP 路径不变。作者 checklist 未勾选单测与精度评估项,正文未附量化精度对比数据。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
