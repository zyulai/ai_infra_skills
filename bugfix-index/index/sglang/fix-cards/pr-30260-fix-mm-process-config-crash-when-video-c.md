# PR #30260: [Fix] --mm-process-config crash when video config contains

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30260](https://github.com/sgl-project/sglang/pull/30260) |
| 作者 | longxin9715 |
| 合并日期 | 2026-07-28 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 22 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
`--mm-process-config` 的视频配置键被原样传给 transformers,`max_frames` 等键触发 typed-dict 校验崩溃;预采样视频过滤 sglang 自有键后再传入修复。

## 问题现象
给 Qwen3-VL 加 `--mm-process-config '{"video":{"min_pixels":1000,"max_pixels":10000}}'` 启动后,带 `video_url` 的 chat 请求直接报错:`TypeError: merged_typed_dict.__init__() got an unexpected keyword argument 'max_frames'`(完整 traceback 经 `serving_chat.py` → `tokenizer_manager.py` → `qwen_vl.py process_mm_data_async` → transformers `processing_utils._merge_kwargs` → `huggingface_hub.dataclasses.validate_typed_dict`)。

## 根因
PR 正文披露:sglang 的视频配置参数被整体传给了 transformers(`The sglang parameter is not passed to transformers` 即参数未按 transformers 预期传递),transformers 的 `_merge_kwargs` 用 typed dict 严格校验 `videos_kwargs`,不认识 `max_frames` 等 sglang 侧键,校验抛 TypeError。由 diff 推断:sglang 在视频已被预采样(`video_metadata` 存在)的场景下,仍把 `fps`/`nframes`/`min_frames`/`max_frames` 等 sglang 自有的预处理配置键塞进 `videos_kwargs`,而 transformers 处理器不接受这些键。

## 修复方案
1. `python/sglang/srt/multimodal/processors/qwen_vl.py`:新增 `QWEN_VIDEO_PREPROCESS_CONFIG_KEYS` 集合(`fps`、`nframes`、`min_frames`、`max_frames`、`min_pixels`、`max_pixels`、`total_pixels`、`resized_height`、`resized_width`)与 `_get_processor_video_config()`——视频已被预采样(全部 metadata 非空)时,把上述 sglang 自有键从 `video_config` 中剔除后再作为 `processor_video_config` 传给处理器;未预采样时返回 None 沿用原配置。
2. `python/sglang/srt/multimodal/processors/base_processor.py` 的 `process_mm_data`:新增可选参数 `processor_video_config`,存在时以它替代 `self.video_config` 注入 `videos_kwargs`。
3. `qwen_vl.py process_mm_data_async`:统一调用路径,把 qwen3-vl/intern 分支的 `video_metadata`/`do_sample_frames=False` 与通用的 `processor_kwargs` 合并传给 `process_and_combine_mm_data_async`。
4. 顺带修复 `hardware_backend/npu/modules/qwen_vl_processor.py`:`patches.view(...)` 改为 `patches.reshape(...)`。
5. 新增单测:预采样配置过滤、processor 报错不重试、qwen 视频配置路由(过滤/不过滤两分支)。

验证:正文给出 Qwen3-VL-8B-Instruct 带 `video_url` 的 curl 请求,修复后正常返回视频理解内容。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:多模态视频处理路径。行为变化仅当视频已预采样且 `--mm-process-config` 视频配置含 sglang 自有键时——这些键不再传给 transformers(消除崩溃),其余键(如 `do_normalize`)仍传;未预采样路径行为不变。`process_mm_data` 新增可选参数对其他处理器无影响;统一后的 `process_and_combine_mm_data_async` 调用在 qwen3-vl 等分支语义等价。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
