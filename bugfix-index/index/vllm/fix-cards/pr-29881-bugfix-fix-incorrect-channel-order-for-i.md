# PR #29881: [Bugfix] Fix incorrect channel order for idefics3 in edge case

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29881](https://github.com/vllm-project/vllm/pull/29881) |
| 作者 | Isotr0py |
| 合并日期 | 2025-12-02 |
| labels | ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
idefics3 图像处理器在边缘情形(如 (3,3,3) 方形图)误判通道顺序;给 HF processor 显式注入 input_data_format=channels_last。

## 问题现象
idefics3 的图像处理器在边缘情形下会错误检测通道顺序(channel order)。正文引用了 transformers 的警告原文:

```
The channel dimension is ambiguous. Got image shape (3, 3, 3). Assuming channels are the first dimension. Use the [input_data_format](https://huggingface.co/docs/transformers/main/internal/image_processing_utils#transformers.image_transforms.rescale.input_data_format) parameter to assign the channel dimension.
```

即当图像各维相等(如 3×3×3)时通道维存在歧义,处理器只能猜测,猜测错误会导致图像被按错误的通道顺序处理,产生错误的预处理结果。

## 根因
正文披露:「Idefics3's image processor will detect channel order incorrectly under edge case」——调用 HF 图像处理时未指定 `input_data_format`,在通道维有歧义的边缘情形(各维尺寸相同的图像)下,HF processor 按默认假设(通道在第一维)处理,与实际数据布局不符。

## 修复方案
改 `vllm/model_executor/models/idefics3.py` 的 `_call_hf_processor`:在调用 `super()._call_hf_processor()` 之前,向 `mm_kwargs` 注入默认值 `input_data_format: channels_last`(写在字典最前,用户传入的 mm_kwargs 可覆盖):

```diff
+        mm_kwargs = {"input_data_format": "channels_last", **mm_kwargs}
         processed_outputs = super()._call_hf_processor(
```

这样 HF processor 不再自行猜测通道维,消除歧义图像上的错误通道顺序。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 idefics3 模型的 `_call_hf_processor` 图像预处理路径;对正常(通道维无歧义)图像,显式 channels_last 与原检测结论一致时结果不变;若下游用户依赖此前的「channels_first 猜测」行为处理非常规数据,行为会改变,但可通过 mm_kwargs 覆盖回原值。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
