# PR #16593: fix: adjusting vlm accuracy thresholds

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16593](https://github.com/sgl-project/sglang/pull/16593) |
| 作者 | dougyster |
| 合并日期 | 2026-01-07 |
| labels | Multi-modal |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
VLM 精度 nightly 测试因多处「近失」而失败;放宽 6 个模型的 score/latency 阈值以减少偶发失败。

## 问题现象
`nightly-test-vlm-accuracy-2-gpu-runner` 上有相当多「near miss」(近失)导致测试失败。merrymercy 要求「reduce the threshold for all failed nightly-test-vlm-accuracy-2-gpu-runner」并附失败 run 链接。涉及调整的模型:zai-org/GLM-4.5V-FP8、moonshotai/Kimi-VL-A3B-Instruct、Qwen/Qwen2.5-VL-7B-Instruct、OpenGVLab/InternVL2_5-2B、unsloth/Mistral-Small-3.1-24B-Instruct-2503(diff 中另有 openbmb/MiniCPM-o-2_6)。

## 根因
正文说明:失败源于测试阈值设置过紧产生的近失,需要放宽阈值以减少这类偶发失败。属于测试判定标准校准,正文未指向任何生产代码缺陷。

## 修复方案
`test/registered/eval/test_vlms_mmmu_eval.py` 中各模型的 `ModelEvalMetrics(score, latency)` 阈值调整(提高 latency 上限、小幅下调 score 下限):
- Kimi-VL-A3B-Instruct:latency 22.3 → 23.5
- MiniCPM-o-2_6:latency 29.3 → 29.5
- InternVL2_5-2B:latency 17.0 → 18.0
- Qwen2.5-VL-7B-Instruct:score 0.340 → 0.330
- Mistral-Small-3.1-24B-Instruct-2503:score 0.310 → 0.30
- GLM-4.5V-FP8:latency 32.0 → 34.0

```python
-    ModelLaunchSettings("moonshotai/Kimi-VL-A3B-Instruct"): ModelEvalMetrics(
-        0.330, 22.3
-    ),
+    ModelLaunchSettings("moonshotai/Kimi-VL-A3B-Instruct"): ModelEvalMetrics(
+        0.330, 23.5
+    ),
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 VLM MMMU 评测测试的判定阈值,不涉及生产代码;阈值放宽后,模型真实精度的轻微下滑更不易被该测试捕获。

## 关联
正文未列出 issue/PR,引用的 CI 链接:
- https://github.com/sgl-project/sglang/actions/runs/20513077000/job/58936524276
- https://github.com/sgl-project/sglang/actions/runs/20764464710/job/59627330523

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
