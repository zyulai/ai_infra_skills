# PR #12240: [VLM] Optimize qwen_vl preprocess_video

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-11-06 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | performance, Multi-modal, run-ci, vlm |
| Diff | 素材未提供 diffstat |
| 链接 | https://github.com/sgl-project/sglang/pull/12240 |

## 一句话结论

多管齐下优化 Qwen-VL 视频预处理,preprocess 总耗时降 38.4%,2MB 视频请求 e2e 提速 14.2%(3.37s→2.89s)。

## 动机

Qwen-VL 的 `preprocess_video` 与后续 multimodal 处理在视频请求中占比很高:2MB 视频打 Qwen2.5-VL-72B(tp8)时,preprocess_video 约 405.7 ms、`process_and_combine_mm_data` 约 156.8 ms。其中 `get_batch`(抽帧解码)约 283.9 ms,`torchvision resize` 约 111.9 ms,均存在明显浪费,值得逐段消除。

## 优化点

五个针对性改动:(1) 抽帧索引用 `np.linspace(..., dtype=np.int64) + np.unique` 去重,替代 `torch.linspace().round()`,避免 rounding 产生重复帧导致的重复解码与内存拷贝;(2) `torch.from_numpy(video_np).pin_memory()` 替代 `torch.tensor(video)`,省一次 CPU 拷贝;(3) resize 插值由 BICUBIC+antialias 换成更快的 BILINEAR 并去掉后面的 `.float()` 转换(曾尝试把 resize offload 到 GPU,但 ≥150MB 大视频会 OOM,最终保留 CPU resize);(4) 让视频特征留在 GPU,配合 `--keep-mm-feature-on-device` 避免 `BaseMultimodalProcessor::process_mm_data` 末尾额外的 D2H 拷贝,HF `process_mm_data` 直接在 GPU 上跑;(5) 加了各阶段耗时日志,并把多个视频的 `asyncio.gather` 并发改为顺序 await(diff 中可见)。

## 关键改动

```text
python/sglang/srt/multimodal/processors/qwen_vl.py: 唯一改动文件——抽帧去重、拷贝消除、BILINEAR resize、perf 日志、顺序化 preprocess_video 调用
```

```python
idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
idx = np.unique(idx)
video_np = vr.get_batch(idx).asnumpy()
video = torch.from_numpy(video_np).pin_memory()
video = video.permute(0, 3, 1, 2)  # Convert to TCHW format
...
video = torchvision.transforms.functional.resize(
    video, [resized_height, resized_width],
    interpolation=InterpolationMode.BILINEAR,
)
video = video.pin_memory()
```

## 性能声明

测试条件:Qwen2.5-VL-72B-Instruct,tp8,2MB 视频推理(`--attention-backend fa3 --mm-attention-backend fa3 --keep-mm-feature-on-device` 等,见正文启动命令)。各阶段:get_batch 283.89→240.82 ms(-15.17%),torchvision resize 111.87→9.02 ms(-91.9%),preprocess 总计 405.72→249.86 ms(-38.4%),process_and_combine_mm_data 156.81→37.96 ms(-75.8%),process_and_collect_mm_items 约 150→约 39 ms;客户端 e2e 3.37s→2.89s(-14.2%)。附服务端逐请求日志佐证,输出文本两版一致。

## 验证与回归风险

未新增自动化测试,以人工 e2e 对比(客户端输出与服务端日志)验证。回归风险:插值算法从 BICUBIC 改为 BILINEAR、抽帧索引去重,都会改变送入模型的像素与帧集合,可能影响视频任务的输出分布;`pin_memory` 增加页锁定内存占用;GPU 常驻路径依赖 `--keep-mm-feature-on-device`,未设置时行为不变(该 flag 缺省仍回拷 CPU)。改动仅限 qwen_vl 处理器,不影响其他模型的多模态路径。

## 关联

无(素材中未引用其他 PR 编号;正文提到 `--keep-mm-feature-on-device` 为既有功能,依赖其避免 D2H 拷贝)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
