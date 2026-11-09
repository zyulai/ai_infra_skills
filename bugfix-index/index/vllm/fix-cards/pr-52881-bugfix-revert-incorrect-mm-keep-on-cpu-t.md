# PR #52881: [BugFix] Revert incorrect MM keep_on_cpu=True changes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52881](https://github.com/vllm-project/vllm/pull/52881) |
| 作者 | njhill |
| 合并日期 | 2026-08-19 |
| labels | bug, ready, qwen |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
多模态字段被错误地设为 keep_on_cpu=True 引发 CI 回归,撤销这些改动并保留 non_blocking 传输修复。

## 问题现象
正文仅说明这是修复由 #52827 引入的 CI 回归("Fix CI regression introduced by #52827"),未描述具体失败现象与触发条件。

## 根因
正文披露的根因:#52827 对多模态(MM)字段所做的 keep_on_cpu=True 改动是不正确的,直接导致 CI 回归;正文未进一步展开失败机制。
从 diff 可以描述其暗示的机制:#52827 曾把多个模型的 grid 尺寸类张量(如 image_grid_thw、tgt_sizes、speech_lengths、patch_positions、feature_attention_mask、audio_feature_lengths 等)改为保留在 CPU;本 PR 撤销其中错误的设置,同时在调用点补上 `non_blocking=True` 的异步传输,并把 ernie45_vl 中 `num_valid` 的求和改回在 CPU 张量上计算,从而避免原本需要的 GPU 同步(由 diff 推断)。

## 修复方案
撤销 6 个模型文件中错误的 keep_on_cpu=True:
- fireredlid.py:`speech_lengths` 恢复为普通 batched 字段(仅 `fake_token_lengths` 保留 keep_on_cpu=True)
- llava_onevision2.py:image/video 的 `patch_positions`、`patch_positions_videos` 恢复进 GPU(仅 `video_num_frames`、`frame_timestamps` 等保留)
- minicpmv4_6.py:`tgt_sizes`、`video_tgt_sizes` 恢复进 GPU(`video_image_sizes` 保留)
- moss_transcribe_diarize.py:`audio_feature_lengths` 恢复进 GPU(`audio_chunk_counts` 保留)
- qwen2_audio.py:`feature_attention_mask` 恢复进 GPU
- ernie45_vl.py / isaac.py:相应的 `.to(device)` 改为 `non_blocking=True`

关键改动示例(ernie45_vl.py,不再需要 gpu_sync_allowed):
```python
grid_thw_cpu = batch_mm_kwargs["image_grid_thw"]
grid_thw = grid_thw_cpu.to(output.device, non_blocking=True)
num_valid = int(
    (grid_thw_cpu[:, 0] * grid_thw_cpu[:, 1] * grid_thw_cpu[:, 2]).sum()
)
```

## 影响与回归风险
- 影响窗口:正文未说明(仅知回归由 #52827 引入、本 PR 于 2026-08-19 合并修复)
- 回归面:ernie45_vl、fireredlid、isaac、llava_onevision2、minicpmv4_6、moss_transcribe_diarize、qwen2_audio 这 7 个多模态模型的输入字段设备布局发生变化:被撤销的字段重新进入 GPU;若下游代码仍假设这些字段在 CPU,可能再次出现设备不匹配。`non_blocking=True` 依赖 pinned memory 的正确性,也是潜在回归点。

## 关联
- [#52827](https://github.com/vllm-project/vllm/pull/52827)(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
