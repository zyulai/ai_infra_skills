# PR #13487: [Performance] Replace preprocess_video logic from GLM  multimodal processor with transformer impl for speed up (up to 27% faster) and addressing OOM (up to 50x improvements)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @byjiang1996 |
| 合并 | 2025-11-25 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | dependencies, run-ci |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 内容可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/13487 |

## 一句话结论

GLM/Qwen3-VL 视频预处理改用 transformers 官方实现:大视频提速最高 27.3%,内存 200GB+→4GB(50x),消除 OOM。

## 动机

GLM 团队反馈:处理大视频(>100MB)时 GLM-4xV 极慢且极易 OOM。根因是 SGLang 在 `glm4v.py` processor 里自实现的 `preprocess_video`(从 transformers video_utils 移植、走 decord `VideoReader` 一次性提取全部帧)不如 transformers 中 model owners 的优化实现(video_processing_glm4v / video_processing_qwen3_vl)。作者的观点是:应尽量复用 transformers 的实现,而不是在 SGLang 里复制一份容易过时、引发隐蔽性能问题的逻辑。

## 优化点

- 删除 `Glm4vMultimodalProcessor.preprocess_video`(decord 全帧提取 + 手工组装 `frames_indices`/fps/duration metadata),`process_mm_data_async` 中直接 `base_output.videos = request_obj.video_data`,把原始视频数据交给 transformers 侧处理;`process_and_combine_mm_data` 不再传 `video_metadata`,帧采样逻辑完全由 transformers 当前版本决定。
- 依赖调整:`python/pyproject.toml` 与 `pyproject_xpu.toml` 新增平台条件依赖——`torchcodec==0.7.0/0.8.0`(非 Linux ARM;PR 注明 torchcodec 不存在于这些平台,缺失时 transformers 默认回退 torchvision)与 `av`(Linux ARM);同时把 diffusion extra 中旧固定的 `torchcodec==0.5.0` 移除以避免版本冲突。

## 关键改动

```text
python/sglang/srt/multimodal/processors/glm4v.py: 删除自实现 preprocess_video,视频数据直传 transformers
python/pyproject.toml: 新增 torchcodec==0.7.0 / av 平台条件依赖,移除 diffusion 的 torchcodec==0.5.0
python/pyproject_xpu.toml: 同步新增 torchcodec==0.8.0 / av 条件依赖
```

## 性能声明

测试条件:PR 用 GLM-4.1V-9B-Thinking(及 Qwen3-VL-8B-Instruct)起服务(`--mm-attention-backend fa3 --attention-backend fa3`),curl 走 `/v1/chat/completions` 发送 531MB 视频(`/tmp/data/1O1TfTrEnss.mp4`,来自 videomme benchmark 的 videos_chunked_01.zip),temperature 0,max_completion_tokens 1000;PR 未注明 CPU/RAM 等硬件配置。

- Qwen3-VL-8B-Instruct:31.2s → 24.5s,快 27.3%。
- GLM-4.1V-9B-Thinking:修改前 OOMKilled(3min20s),修改后 27.5s 完成。
- 内存压力:处理同一视频的内存消耗从 200GB+ 降到 4GB,PR 称 "up to 50x improvements"。

## 验证与回归风险

PR 附 GLM41v 与 Qwen3-VL 的 UT 通过截图(具体测试名未在 pack 中列出)。风险:视频采样行为完全交给 transformers 版本,升级 transformers 会改变帧数/fps 处理(与原 decord 全帧提取的语义可能有差异);新增 torchcodec/av 平台条件依赖,ARM Linux 上回退 torchvision 路径未在 PR 中单独验证;删除的 metadata 组装逻辑(`video_metadata=[...]` 嵌套格式)若有下游依赖需 transformers 侧兼容。

## 关联

无(pack 未给出同区域其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
