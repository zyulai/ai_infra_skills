# PR #23366: [BugFix] Fix the issue where image embeddings were incorrectly split.…

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23366](https://github.com/vllm-project/vllm/pull/23366) |
| 作者 | bppps |
| 合并日期 | 2025-08-22 |
| labels | ready, qwen |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Qwen2-VL/Qwen2.5-Omni 的 DataParser 按 `image_grid_thw` 全乘积切分 image/video embeddings,未除以 `spatial_merge_size`²,导致切分越界;改为按空间合并后的尺寸切分。

## 问题现象
当用户直接提供 `image_embed` 和 `image_grid_thw` 时,image embeddings 会被按 `image_grid_thw` 的乘积错误切分,而 embeddings 实际是经过 `spatial_merge_size` 空间合并后的长度,于是切分越界(正文原话:"slice error (out of range without merging)");视频 embedding 存在同样问题。Test Plan 描述的触发场景:多次输入多个 image embeddings、且上下文长度未与 KVCache block size 对齐时,最后一张图会在 prefill 阶段重算,此时会因先前切分越界出现维度不匹配(dimension mismatch)。对应 issue #23371。

## 根因
正文披露:切分本应按 `image_grid_thw` 乘积除以 `spatial_merge_size` 后的数量进行,这一逻辑在 Qwen2-VL 模型的 input embedding 处理流程中已经存在(正文给出 qwen2_vl.py 与 qwen2_5_omni_thinker.py 的空间合并代码引用),但 Qwen2_VL / Qwen2_5_Omni 的 MultiModalDataParser 在用户直接传 `image_embed` + `image_grid_thw` 的场景下「may not be considered carefully」,仍按未合并的乘积切分,造成越界。

## 修复方案
改了 3 个文件:
- `vllm/model_executor/models/qwen2_vl.py`:把 `_qwen2vl_field_config` 重构为工厂 `_create_qwen2vl_field_factory(spatial_merge_size)`;内部 `pixel_values` 仍按 `image_grid_thw.prod(-1)`(pixel grid sizes)切分,而 `image_embeds` 改用 `// spatial_merge_size // spatial_merge_size` 后的 embed grid sizes;视频同理(`video_embed_grid_sizes`)。`Qwen2VLMultiModalDataParser` 构造函数新增 `spatial_merge_size` 参数,`_parse_image_data`/`_parse_video_data` 的 `fields_factory` 传入该值;`_get_data_parser` 用 `vision_config.spatial_merge_size` 构造 parser。
- `vllm/model_executor/models/qwen2_5_omni_thinker.py`:同样改造为 `create_qwen2_5_omni_thinker_field_factory(spatial_merge_size)`,parser 与 `_get_mm_fields_config` 均传入 `vision_config.spatial_merge_size`。
- `vllm/model_executor/models/glm4_1v.py`:`_get_mm_fields_config` 改用带 `spatial_merge_size` 的 `_create_qwen2vl_field_factory`。

关键 diff(以 qwen2_vl.py 为例):

```diff
+        image_pixel_grid_sizes = image_grid_thw.prod(-1)
+        image_embed_grid_sizes = (image_pixel_grid_sizes //
+                                  spatial_merge_size // spatial_merge_size)
...
-        image_embeds=MultiModalFieldConfig.flat_from_sizes(
-            "image", image_grid_sizes),
+        image_embeds=MultiModalFieldConfig.flat_from_sizes(
+            "image", image_embed_grid_sizes),
```

## 影响与回归风险
- 影响窗口:正文未提及具体版本/提交范围,不写。
- 回归面:Qwen2-VL、Qwen2.5-Omni(thinker)与 GLM-4.1V 三个模型的多模态数据解析与字段切分路径;`pixel_values` 的切分尺寸语义未变,`image_embeds`/`video_embeds` 在 spatial_merge_size=1 的模型上数值不变(除以 1),在合并系数大于 1 的模型上切分长度变小——这正是修复目标。fields factory 由模块级函数改为闭包工厂,相关 processor 的调用路径均已在本次 diff 中同步更新。

## 关联
- FIX #23371: https://github.com/vllm-project/vllm/issues/23371
- image embedding spatial merge 代码引用: qwen2_vl.py#L1179, qwen2_5_omni_thinker.py#L656
- video embedding spatial merge 代码引用: qwen2_vl.py#L1197, qwen2_5_omni_thinker.py#L676

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
