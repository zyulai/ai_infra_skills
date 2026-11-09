# PR #36206: [CI][MM] Gate vision encoder attention mask to MiniCPM only, fixing Aria regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36206](https://github.com/vllm-project/vllm/pull/36206) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-03-06 |
| labels | ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
#34127 在共享视觉编码器中加入注意力掩码传播,殃及不做掩码的 Aria 造成输出偏差;新增开关仅让 MiniCPM 启用掩码。

## 问题现象
Aria 模型出现输出偏差(output divergence),多模态测试 `test_multi_image_models[aria-test_case95]` 失败。触发条件:PR #34127 合并后,凡是复用共享 `Idefics2VisionTransformer` 编码器、但自身视觉注意力层不使用掩码的模型(如 Aria)都会受影响。正文未提供具体数值差异日志。

## 根因
正文披露:PR #34127 为修复 MiniCPM-o 4.5,在共享的 `Idefics2VisionTransformer` 视觉编码器中加入了注意力掩码传播。但该编码器同时被 Aria(和 Phi4MM)使用,这些模型的视觉注意力层并不应用掩码,只在 embedding 阶段使用 `patch_attention_mask`;共享编码器中被强制构造并传入的加性掩码改变了它们的前向输出,形成回归。

## 修复方案
- `vllm/model_executor/models/idefics2_vision_model.py`:`Idefics2VisionTransformer` 新增 `apply_encoder_attention_mask` 参数(默认 `False`)并在 `__init__` 保存;`forward` 中把掩码构造条件从"有 padding 即构造加性 4D 掩码"改为"先检查开关,未开启则直接跳过":
```diff
-        if flat_patch_mask is None or not torch.any(~flat_patch_mask):
+        if (
+            not self.apply_encoder_attention_mask
+            or flat_patch_mask is None
+            or not torch.any(~flat_patch_mask)
+        ):
             attention_mask = None
```
- `vllm/model_executor/models/minicpmv.py`:v2.5、v2.6、v4.0、v4.5 四处 `init_vision_module` 构造 `Idefics2VisionTransformer` 时显式传入 `apply_encoder_attention_mask=True`。
- `aria.py`、`phi4mm.py` 无任何改动,沿用默认 `False`,恢复无掩码行为。

## 影响与回归风险
- 影响窗口:#34127 合并之后至本 PR 之前,使用共享编码器的 Aria(以及潜在 Phi4MM)输出异常。
- 回归面:所有 `Idefics2VisionTransformer` 消费者。MiniCPM 系列显式 opt-in,掩码行为不变;Aria/Phi4MM 及其他默认消费者恢复到 #34127 之前的无掩码前向。今后若有新模型确实需要编码器级掩码,必须显式传 `apply_encoder_attention_mask=True`,否则会静默回到不掩码路径。

## 关联
- PR #34127(引入回归的改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
