# PR #6367: [Bugfix] Fix dtype mismatch in PaliGemma

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6367](https://github.com/vllm-project/vllm/pull/6367) |
| 作者 | DarkLight1337 |
| 合并日期 | 2024-07-12 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

PaliGemma 的 pixel_values 未按视觉塔权重的 dtype 转换,半精度下 dtype 不匹配;进塔前显式 cast 到目标 dtype 并补 half 测试。

## 问题现象

正文未描述具体现象,仅写 "FIX #6366"。从 diff 可见的触发场景(由 diff 推断):非默认精度(如 half/float16)下运行 PaliGemma——diff 同时把 `tests/models/test_paligemma.py` 的 `dtype` 参数从 `["float"]` 扩展为 `["float", "half"]`,补上半精度回归覆盖;修复前该 dtype 组合下视觉塔前向会因输入张量与权重的 dtype 不一致而出错。

## 根因

正文未披露根因(正文只有 "FIX #6366" 一句与 issue 链接)。从 diff 描述改动所暗示的机制(由 diff 推断):`_image_pixels_to_features` 直接把 `pixel_values` 传入 `vision_tower`,未做任何 dtype 转换;当模型以 half 运行时,图像像素张量与 SiglipVisionModel 权重的 dtype 不匹配,前向计算抛出 dtype 不一致的错误。

## 修复方案

1. `vllm/model_executor/models/paligemma.py` 核心修复:`_image_pixels_to_features` 在调用视觉塔前读取目标 dtype 并显式转换像素张量:

```diff
-        image_outputs = vision_tower(pixel_values, output_hidden_states=True)
+        target_dtype = vision_tower.get_input_embeddings().weight.dtype
+        image_outputs = vision_tower(pixel_values.to(dtype=target_dtype),
+                                     output_hidden_states=True)
```

2. 同文件顺手修了一个潜在崩溃点:`input_processor_for_paligemma` 中把 `if image_token_str in orig_prompt:` 改为 `if orig_prompt is not None and image_token_str in orig_prompt:`,避免 prompt 为 None 时 `in` 操作报错。
3. 接口对齐:`paligemma.py` 与 `gemma.py` 的 `forward` 增加 `intermediate_tensors: Optional[IntermediateTensors] = None` 参数(gemma 在 `inputs_embeds` 前插入该参数,调用处传 `None`),并导入 `IntermediateTensors`。
4. `tests/models/test_paligemma.py`:dtype 参数化增加 `"half"`,使半精度路径进入常规模型测试矩阵。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,影响所有 PaliGemma 多模态输入路径:pixel_values 现在总是被 cast 到视觉塔嵌入权重 dtype,默认 float 路径行为不变,其他精度路径由报错变为正常执行;`forward` 签名新增带默认值的 `intermediate_tensors` 参数,自定义调用方需按新位置传参(默认 None 兼容);prompt 为 None 的输入分支不再异常。Gemma 纯文本模型仅签名对齐,计算逻辑未变。

## 关联

- https://github.com/vllm-project/vllm/issues/6366 (FIX)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
