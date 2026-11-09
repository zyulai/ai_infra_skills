# PR #44509: [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44509](https://github.com/vllm-project/vllm/pull/44509) |
| 作者 | tc-mb |
| 合并日期 | 2026-06-04 |
| labels | bug, multi-modality, ready |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
MiniCPM-V-4.6 视频请求因帧尺寸解析错乱致占位符数与视觉嵌入数不符、引擎崩溃;三处修复对齐计数。

## 问题现象
向 `openbmb/MiniCPM-V-4_6` 发送视频请求会触发 `EngineDeadError`——engine core 崩溃,因为 `<|video_pad|>` 占位符 token 数与 vision tower 产出的视觉嵌入数不一致,`_merge_multimodal_embeddings` 抛 `ValueError`。图像推理正常,仅视频触发。验证环境:2× NVIDIA 4090、`tensor-parallel-size 2`、vLLM 0.22.0 打上本修复,经 `/v1/chat/completions` 发送 4 秒 320×240 视频可正常返回描述文本。

## 根因
正文披露三个问题叠加:
1. `VideoProcessorItems.get_frame_size`(`parse.py`)硬编码按 `(C, H, W)` 解包;`_get_video_with_metadata` 把 PIL 图片列表经 `np.array(...)` 转成 `(N, H, W, C)`(HWC),旧代码把 channel 维当宽、高维当宽,返回错误的 `ImageSize(3, W)` 而非 `ImageSize(W, H)`;
2. `process_videos`(`minicpmv4_6.py`)逐帧手调 image processor,只返回 `video_pixel_values` + `video_tgt_sizes`,从不产出上游 `MiniCPMVMultiModalProcessor.process_videos` 通常会提供的 `video_image_sizes`;
3. `_get_prompt_updates` 从原始 `mm_items`(`VideoProcessorItems`)读帧尺寸/帧数,拿不到已处理数据,占位符计数于是被问题 (1) 的错误帧尺寸驱动。
结果:占位符数 ≠ 实际嵌入数。

## 修复方案
2 个文件 +64/−2(纯 Python,无需重编 C++/CUDA):
- `minicpmv4_6.py`:`process_videos` 逐帧记录 `(W, H)` 并返回 `video_image_sizes`;`get_video_replacement` 优先用 `out_mm_kwargs` 里处理后的 `video_image_sizes` 驱动占位符计数,拿不到时回退旧路径;
- `parse.py`:`get_frame_size` 先判别 HWC/CHW 再解包:

```python
if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
    # HWC format (e.g. from np.array(PIL.Image) via
    # _get_video_with_metadata).  PIL images are always
    # channels-last.
    h, w = image.shape[0], image.shape[1]
else:
    # CHW format (standard PyTorch / numpy convention).
    _, h, w = image.shape
```

## 影响与回归风险
- 影响窗口:正文明确"不是回归——MiniCPM-V-4.6 的视频路径在 vLLM 上从未工作过",即自该模型支持引入起即受影响。
- 回归面:`parse.py` 的 `get_frame_size` 是通用多模态解析路径,所有经 `VideoProcessorItems` 取帧尺寸的调用方都走新判别逻辑(shape[-1]∈{1,3,4} 判 HWC);minicpmv4_6 视频路径新增优先数据源但保留回退。正文未列专门单测,验证为手测。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
