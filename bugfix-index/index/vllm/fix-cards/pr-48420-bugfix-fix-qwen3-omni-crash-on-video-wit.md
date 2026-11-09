# PR #48420: [Bugfix] Fix Qwen3-Omni crash on video with no audio track when use_audio_in_video=True

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48420](https://github.com/vllm-project/vllm/pull/48420) |
| 作者 | RyanJHamby |
| 合并日期 | 2026-08-03 |
| labels | bug, multi-modality, qwen, ready |
| 热度 | 👍 3 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | crash |

## 一句话结论
无音轨视频配 use_audio_in_video=True 时 HF processor 抛 StopIteration 崩溃;vLLM 侧检测到无音频即抛明确 ValueError,不再转发该标志。

## 问题现象
调用方可以在不知道视频是否带音轨的情况下请求 `use_audio_in_video=True`;当视频没有音轨时,Qwen3-Omni 处理直接崩溃。作者用与 issue 报告相同形状 `(16, 3, 480, 256)` 的合成无音轨视频直接复现,报错为:

```
File ".../transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py", line 247, in replace_multimodal_special_tokens
    audio_token_indices = np.arange(next(audio_lengths))
StopIteration
```

并确认同一 processor 用 `use_audio_in_video=False` 调用则正常产出 `input_ids`、`attention_mask`、`pixel_values_videos`、`video_grid_thw`、`video_second_per_grid`。

## 根因
正文披露(经复现验证):视频无音轨时不会从中提取任何 audio item,但 `Qwen3OmniMoeThinkerProcessingInfo._call_hf_processor` 仍无条件把 `use_audio_in_video=True` 转发给底层 HF `Qwen3OmniMoeProcessor`;该 HF processor 假设此标志意味着每个视频至少有一个 audio item,在假设不成立时于 `replace_multimodal_special_tokens` 中对耗尽的迭代器执行 `next(audio_lengths)`,抛出未防护的 `StopIteration`;vLLM 的 `InputProcessingContext.call_hf_processor` 又把它包成笼统的 `ValueError`,即 #48383 报告的现象。

## 修复方案
按 diff 实际改动(注:正文 Fix 一节描述为「回退 use_audio_in_video=False」,最终 diff 的做法是检测到无音频时抛出明确错误,不再让 HF processor 抛 StopIteration):
- `vllm/model_executor/models/qwen3_omni_moe_thinker.py` 的 `_call_hf_processor`:当 `use_audio_in_video` 为真且 `mm_data` 里有 videos 时,抛出 `ValueError("Video doesn't have audio track with \`audio_in_video=True\`")`;注释说明 mm_data 为空可能是多模态 processor cache 命中,不算真正的无音频场景。
- `vllm/model_executor/models/qwen2_5_omni_thinker.py`:`_call_hf_processor` 增加类似防护(并考虑子类已自行填入 audio、cache 命中等非真「无音频」情形);`_apply_hf_processor_mm_only` 把裸 `assert "audio" in mm_counts` 改为当 audio 数少于 video 数时抛同样的明确 ValueError。
- `tests/models/multimodal/processing/test_audio_in_video.py`:新增 `test_use_audio_in_video_without_audio_track`,对 Qwen2.5-Omni-3B 与 Qwen3-Omni-30B-A3B-Instruct 断言抛出含 "doesn't have audio track" 的 ValueError。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:Qwen2.5-Omni / Qwen3-Omni thinker 的多模态预处理路径;此前「无音轨视频 + use_audio_in_video=True」会得到 HF 的 StopIteration(被包成笼统 ValueError),现在改为明确的业务 ValueError,报错语义变化;cache 命中与子类预填 audio 的场景被显式排除,不影响正常请求。

## 关联
- Fixes [#48383](https://github.com/vllm-project/vllm/issues/48383)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
