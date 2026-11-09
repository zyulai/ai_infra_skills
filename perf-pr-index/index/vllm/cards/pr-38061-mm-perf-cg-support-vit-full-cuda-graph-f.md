# PR #38061: [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-04-14 |
| 分类 | CUDA Graph / Compiler（tier: major） |
| Label | documentation, performance, ready, v1, multi-modality, qwen, nvidia |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾标注 1 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/38061 |

## 一句话结论

Qwen3-VL 视频启用 ViT full CUDA graph，1×A100/FLASHINFER P99 由 58.62ms 降至 7.27ms(+87.60%)。

## 动机

正文 Purpose 与 AI Summary：承接 #35963（仅支持 image inference 的 ViT full CUDA graph），本 PR 继续为 Qwen3-VL 支持 video inference。此前 CUDA graph 的 capture/replay 路径只处理 image 输入（`pixel_values` + `image_grid_thw`）；video 输入使用不同 key（`pixel_values_videos` + `video_grid_thw`），且因每个 video item 贡献多帧（T 条 attention sequences）需要更大的 `cu_seqlens` 缓冲。本 PR 泛化 protocol 与 manager，用单一共享 graph manager 同时处理两种模态。TODO 列出单测、E2E 功能测试、no-DP/DP ViT 下 eager 与 cuda graph 的 benchmark、以及文档更新。

## 优化点

`EncoderCudaGraphConfig`（`encoder_cudagraph_defs.py`）把单个 `input_key` 换成 `input_key_by_modality` dict（如 `{"image": "pixel_values", "video": "pixel_values_videos"}`），按模态路由输入张量。`SupportsEncoderCudaGraph` protocol（`interfaces.py`）新增 `get_input_modality(mm_kwargs)`，并给 `prepare_encoder_cudagraph_capture_inputs()` / `prepare_encoder_cudagraph_replay_buffers()` 增加 `max_frames_per_batch` 参数。`Qwen3VLForConditionalGeneration`（`qwen3_vl.py`）实现 `get_input_modality()`、新增 `_get_pixel_values_by_modality()` / `_get_grid_thw_by_modality()`，capture 在 `max_frames_per_batch` 超过 `max_batch_size` 时构建 video-format grid（每 item T>1）并为 replay 调整 `cu_seqlens` 缓冲，新增按 `(modality, grid_thw)` 缓存的 `_replay_buffer_cache` 避免重复 CPU 侧 NumPy 计算，`prepare_encoder_metadata()` 接受 `max_frames_per_batch` 做 padding。`EncoderCudaGraphManager`（`encoder_cudagraph.py`）为 `BudgetGraphMetadata` 与初始化加入 `max_frames_per_batch`，capture 用 image dummy 输入同时覆盖 image/video（两者 per-patch shape 相同），replay 时经 `get_input_modality()` 选 key。`CompilationConfig`（`compilation.py`）新增 `encoder_cudagraph_max_frames_per_batch`（0 = auto-infer）并重命名原 `encoder_cudagraph_max_images_per_batch`——diff 中实际名为 `encoder_cudagraph_max_vision_items_per_batch`，正文 AI Summary 与 benchmark 命令记作 `encoder_cudagraph_max_mm_items_per_batch`。EVS（Efficient Video Sampling）pruning 启用时 video CUDA graph 自动禁用（token 数 data-dependent，不兼容 capture）。

## 关键改动

```text
vllm/v1/worker/encoder_cudagraph_defs.py: EncoderCudaGraphConfig 用 input_key_by_modality 替换单 input_key
vllm/model_executor/models/interfaces.py: SupportsEncoderCudaGraph 新增 get_input_modality() 与 max_frames_per_batch 参数
vllm/model_executor/models/qwen3_vl.py: 实现模态路由、video grid/cu_seqlens padding 与 replay buffer 缓存
vllm/v1/worker/encoder_cudagraph.py: manager 加入 max_frames_per_batch，replay 按模态选 input_key
vllm/config/compilation.py: 新增 encoder_cudagraph_max_frames_per_batch 配置并重命名每批上限

# 机制摘录 (encoder_cudagraph.py: replay 路由 + defs.py: 映射)
        input_key = self.config.input_key_by_modality[
            self.model.get_input_modality(mm_kwargs)
        ]
        src = mm_kwargs[input_key]
    input_key_by_modality: dict[str, str]
    """Per-modality input tensor key mapping, e.g.
    {"image": "pixel_values", "video": "pixel_values_videos"}.
    """
```

## 性能声明

Benchmark 用 `vllm bench mm-processor`、random-mm、100 prompts、`(224,224,8)` bucket、seed 42、`cudagraph_mm_encoder: true`、token_budgets 列表、`encoder_cudagraph_max_frames_per_batch: 32`。Single GPU（Qwen3-VL-8B-Instruct，1×A100，video 1 item）：FLASH_ATTN Mean `-24.52% (3.67ms -> 4.57ms)`、P99 `+61.66% (17.03ms -> 6.53ms)`；FLASHINFER Mean `+21.84% (8.38ms -> 6.55ms)`、P99 `+87.60% (58.62ms -> 7.27ms)`。Multi GPU（Qwen3-VL-32B-Instruct，4×A100，TP4、`mm-encoder-tp-mode data`、video 4 items）：FLASH_ATTN Mean `+13.44% (5.43ms -> 4.70ms)`、P99 `+83.22% (51.25ms -> 8.60ms)`；FLASHINFER Mean `+21.37% (8.75ms -> 6.88ms)`、P99 `+82.85% (77.77ms -> 13.34ms)`。以上百分比与括号内 `before -> after` 数值均照抄 PR 正文表格，正文未进一步解释符号约定。

## 验证与回归风险

Unit test `pytest tests/v1/cudagraph/test_encoder_cudagraph.py -v` 结果 36 passed, 3 warnings in 10.04s；Functional test 用 `python examples/offline_inference/vision_language.py -m qwen3_vl --modality "video"` 输出视频描述文本；Benchmark 分单 GPU 与多 GPU 两组命令。新增测试 `SimpleMockViTVideoModel`（双模态）、`TestGetInputModality`（无 GPU）、`TestEncoderCudaGraphVideoReplay`（GPU），覆盖 video capture/replay、fallback、counters、chunking、mixed image+video 共享 manager（+316 lines），并更新 `docs/design/cuda_graphs_multimodal.md`。回归风险：EVS pruning 启用时 video CUDA graph 自动禁用；`SupportsEncoderCudaGraph` protocol 接口新增方法与参数、compilation 配置项被重命名；正文未给出其它模型或硬件的回归评估结论。

## 关联

PR 正文明确承接 #35963（仅支持 image inference 的 ViT full CUDA graph），本 PR 扩展到 video inference；其它同分类 vLLM PR 编号未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
