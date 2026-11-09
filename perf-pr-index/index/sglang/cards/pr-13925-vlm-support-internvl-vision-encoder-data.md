# PR #13925: [VLM] Support InternVL Vision Encoder Data Parallelism

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-11-26 |
| 分类 | Parallelism / Communication(major) |
| Label | feature, performance, Multi-modal, run-ci, vlm |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/13925 |

## 一句话结论

为 InternVL 视觉编码器新增数据并行,把图像 encode 计算分摊到各 TP rank 以降 TTFT,E2E 预期提升 5-10%。

## 动机

源自 issue #12971。profiling 显示 InternVL 的 Vision Encoder 是计算密集型,CUDA command buffer 经常打满;而 TP 下 ViT 内部各 rank 协同算同一批图像。将 vision encoder 改为数据并行,把待编码的图像 chunk 均匀分发到各 rank 独立计算,即可摊薄单 rank 的编码耗时,降低多图请求的 TTFT。

## 优化点

新增 `--mm-enable-dp-encoder` 开关(InternVLChatModel 从 global server args 读取)。DP 模式下:视觉侧 MLP 的 `InternMLP` 用 `tp_size=1 / tp_rank=0` 构造 ColumnParallelLinear/RowParallelLinear(权重不切分,每 rank 完整),attention 同样传 `use_data_parallel`;前向走 `run_dp_sharded_vision_model`(改自 vLLM):把 image_input 第一维 pad 到 `world_size x ceil(chunks/world_size)`,按 TP rank 切片,各 rank 独立跑完整 encoder,再 `tensor_model_parallel_all_gather` 沿 dim 0 收回并裁掉 padding。DP 分支下 encoder 输出不带 hidden_states/attentions,只返回 last_hidden_state。非 DP 路径行为不变;MLP 从 `nn.Linear + ACT2FN` 换成 SGLang 的 parallel linear + `get_act_fn`,以支持自定义 tp_size/tp_rank。

## 关键改动

```text
python/sglang/srt/models/internvl.py: use_data_parallel 贯穿 InternVisionModel/Encoder/Attention/MLP,DP 下 tp_size=1
python/sglang/srt/multimodal/mm_utils.py: 新增 run_dp_sharded_vision_model(pad→per-rank 切片→all_gather)
test/nightly/test_encoder_dp.py: MODELS 增加 OpenGVLab/InternVL2_5-8B(mmmu 阈值 0.52)
```

```python
def run_dp_sharded_vision_model(image_input, vision_model):
    num_chunks = image_input.shape[0]
    mp_world_size = get_tensor_model_parallel_world_size()
    num_chunks_per_rank = (num_chunks + mp_world_size - 1) // mp_world_size
    num_padded_chunks = num_chunks_per_rank * mp_world_size - num_chunks
    pad = (0,) * (2 * (image_input.dim() - 1)) + (0, num_padded_chunks)
    image_input_padded = torch.nn.functional.pad(image_input, pad)
    rank = get_tensor_model_parallel_rank()
    image_input_per_rank = image_input_padded[
        rank * num_chunks_per_rank : (rank + 1) * num_chunks_per_rank, ...
    ]
    vision_embeddings = vision_model(image_input_per_rank)
    ...
    vision_embeddings = tensor_model_parallel_all_gather(vision_embeddings, dim=0)
```

## 性能声明

PR 正文仅给出一句话:"Tested with Vision Encoder DP enabled and Piecewise CUDA Graph enabled. E2E performance is expected to improve 5-10%"——为预期值,未给出具体吞吐/TTFT 数字、模型、硬件与并发条件;Benchmarking 章节为空,作者注明"More benchmark test is in progress"。

## 验证与回归风险

精度用 lmms_eval `mmmu_val` 验证:开启 DP encoder + piecewise cuda graph 的 server 启动配置下,InternVL2_5-8B 得 0.5289(900 题);PR 正文中"without PR"对照组实际跑的是 Qwen2.5-VL-7B-Instruct(同为 0.5289),并非 InternVL 关闭 DP 的对照,严格 A/B 缺失。nightly `test_encoder_dp.py` 把 InternVL2_5-8B 加入与 Qwen2.5-VL-72B 并列的 encoder DP 测试(mmmu_accuracy ≥ 0.52)。风险:非量化权重路径的 `quant_config=None` Column/RowParallelLinear 改写需保证 checkpoint 加载兼容;all_gather 假设各 rank 输出 token 数一致(靠 padding);DP 分支丢弃 hidden_states/attentions,依赖这些输出的调用路径不可用;仅 InternVL 模型接线,其他 VLM 走已有的 `run_dp_sharded_mrope_vision_model`。

## 关联

对应 issue #12971;同文件已有 `run_dp_sharded_mrope_vision_model`(Qwen 系 VLM 的 encoder DP,改编自 vLLM vision.py),本 PR 复用其模式。pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
