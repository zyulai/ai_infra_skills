# PR #19402: Fix nightly Mistral-Large-3 NVFP4 accuracy threshold

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19402](https://github.com/sgl-project/sglang/pull/19402) |
| 作者 | alisonshao |
| 合并日期 | 2026-02-26 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

Mistral-Large-3 的 NVFP4 量化变体 GSM8K 准确率 ~0.875 且不稳定,间歇性跌破 0.90 阈值;nightly 基线阈值从 0.90 降到 0.85。

## 问题现象

Mistral-Large-3 nightly 测试间歇性失败:NVFP4 量化变体(`Mistral-Large-3-675B-Instruct-2512-NVFP4`)的 GSM8K 准确率约为 0.875 且略有不稳定,时常低于 0.90 的基线阈值而误报。同一测试中的 FP8 变体(TP8 与 TP8+MTP)得分为 0.905-0.925,始终在新阈值之上,未受影响。失败示例:[run 22422538348 job 64923364856](https://github.com/sgl-project/sglang/actions/runs/22422538348/job/64923364856#step:4:6868)。

## 根因

正文披露:NVFP4 量化变体的准确率本身在 ~0.875 水平且不稳定,而测试基线阈值统一设为 0.90,导致该变体间歇性低于阈值、nightly 报警。修复是针对 NVFP4 变体的实际精度水平将 gsm8k 基线阈值从 0.90 下调至 0.85。

## 修复方案

改 `test/registered/8-gpu-models/test_mistral_large3.py`,把 `AccuracyTestParams` 的 `baseline_accuracy` 从 0.90 改为 0.85:

```diff
-            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.90),
+            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.85),
```

仅调整 nightly 判定阈值,不改任何产品代码。

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 `test_mistral_large3.py` 的准确率判定阈值。阈值下调 0.05 后,0.85-0.90 区间的真实精度回归不再被拦截,漏报窗口变大;FP8 变体(0.905、0.925)仍高于新阈值,原有通过情况不变。模型推理代码无改动。作者已在 Test plan 中验证:NVFP4(0.875)过新阈值、FP8 变体仍通过。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
