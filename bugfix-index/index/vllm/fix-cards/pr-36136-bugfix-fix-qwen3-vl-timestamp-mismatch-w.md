# PR #36136: [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36136](https://github.com/vllm-project/vllm/pull/36136) |
| 作者 | weiguangli-io |
| 合并日期 | 2026-03-11 |
| labels | bug, multi-modality, qwen, ready |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Qwen3-VL 只传 num_frames 不传 fps 时时间戳数与实际帧数不一致并断言崩溃;改为 num_frames 直接采信、与 HF 对齐。

## 问题现象
通过 `mm_processor_kwargs` 传入 `num_frames` 但不传 `fps` 时,`_get_video_second_idx` 计算出的时间戳长度与实际 `video_grid_thw` 帧数不一致,处理视频即触发断言:

```
AssertionError: The timestamps length(10) should be equal video length (16).
```

影响所有按 num_frames 采样视频的 Qwen3-VL 请求(该修复同样适用于继承 `Qwen3VLProcessingInfo` 的 Qwen3.5-VL)。

## 根因
正文披露:vLLM 侧 `_get_video_second_idx` 在只有 `num_frames` 时回退到默认 fps(2)来反推帧数,而 HF 的 `Qwen3VLVideoProcessor.sample_frames` 把 `num_frames` 与 `fps` 视为互斥——给了 `num_frames` 就直接使用。两边帧数口径不同,导致时间戳条目数与 `video_grid_thw` 帧数不匹配。

## 修复方案
`vllm/model_executor/models/qwen3_vl.py`:
- `_get_video_second_idx` 新增 `sampled_num_frames` 参数;显式给出 `num_frames` 时直接采用,不再由 fps 推算(镜像 HF 行为);
- `_call_hf_processor` 把 `mm_processor_kwargs` 里的 `num_frames` 透传进去,并在指定 `num_frames` 未指定 `fps` 时显式置 `fps=None`,防止 HF `BaseVideoProcessor.preprocess()` 经 `setdefault()` 填入类默认 fps=2 与 `num_frames` 冲突:

```python
if sampled_num_frames is not None:
    num_frames = sampled_num_frames
else:
    sampled_fps = sampled_fps if sampled_fps else video_processor.fps
    num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)
```

新增 `tests/models/multimodal/processing/test_qwen3_vl.py` 回归测试(num_frames=8/16,刻意偏离默认 fps 路径会算出的值)。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:仅 Qwen3-VL 系(`Qwen3VLProcessingInfo` 及其继承者 Qwen3.5-VL)的视频采样路径;基于 fps 的既有采样不变,同时传 `num_frames` 与 `fps` 仍在 HF processor 处报错(既有行为,未改)。

## 关联
- Fixes [#35909](https://github.com/vllm-project/vllm/issues/35909)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
