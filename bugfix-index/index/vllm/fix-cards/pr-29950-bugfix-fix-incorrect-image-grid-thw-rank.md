# PR #29950: [Bugfix] Fix incorrect `image_grid_thw` rank for HunyuanOCR from missing `merge_by_field_config=True`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29950](https://github.com/vllm-project/vllm/pull/29950) |
| 作者 | Isotr0py |
| 合并日期 | 2025-12-03 |
| labels | documentation, ready |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

HunyuanOCR 类缺少 `merge_by_field_config=True`,多图合并后 `image_grid_thw` rank 不正确导致报错;在 HunYuanVLForConditionalGeneration 上补上该类属性修复。

## 问题现象

HunyuanOCR 处理多图输入时出错。复现方式是 `examples/offline_inference/vision_language_multi_image.py` 的多图离线推理(EngineArgs 加载 HunyuanOCR,`limit_mm_per_prompt={"image": N}`,对每张图用不同尺寸的 cherry_blossom 测试图);修复前该复现示例无法工作(正文未给出具体报错信息),修复后可正常运行。

## 根因

正文披露:HunyuanOCR-VL 缺少 `merge_by_field_config=True` 设置,导致 `image_grid_thw` 没有正确的 rank。即该字段作为 multimodal CPU 字段在多图合并(merge)时未按字段配置对齐处理,合并后的张量维数(rank)不符合模型预期。

## 修复方案

修改 `vllm/model_executor/models/hunyuan_vision.py` 的 `HunYuanVLForConditionalGeneration` 类:补充类属性 `merge_by_field_config = True`(紧邻已有的 `multimodal_cpu_fields = {"image_grid_thw"}`)。关键 diff:

```diff
+    merge_by_field_config = True
     multimodal_cpu_fields = {"image_grid_thw"}
```

同时在 `examples/offline_inference/vision_language_multi_image.py` 中新增 HunyuanOCR 的 `load_hunyuan_vl` 加载函数与注册项,作为复现/验证示例。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:改动仅两处——模型类的布尔类属性与示例脚本新增条目;影响面集中在 HunyuanOCR(由 HunYuanVLForConditionalGeneration 承载)的多图预处理路径。该属性只改变 `image_grid_thw` 等字段在多模态合并时的 rank 处理,不影响其他模型;单图输入行为不受影响。

## 关联

- Fix #29812

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
