# PR #22742: [FEAT] [Performance] Enable DP for ViT in Qwen2.5VL

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tjtanaa |
| 合并 | 2025-08-19 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | performance, ready, ci/build, multi-modality, qwen |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/22742 |

## 一句话结论

Qwen2.5-VL 的 ViT 编码器在 TP 组内改数据并行（LLM 仍 TP），VisionArena 基准请求吞吐 +47%、Mean TTFT −55%。

## 动机

Qwen2.5-VL 的 vision encoder（含 Conv3d patch embedding）此前与 LLM 一样按 TP 切分：TP8 下每张卡只算 ViT 的 1/8，多卡间通信/同步开销大而 ViT 计算本身可天然按图片分片。把 ViT 改为 TP 组内数据并行（每卡完整跑一部分图片），可以消除 ViT 的 TP 通信，作者指出大部分收益来自 Conv3d 的 DP 化。

## 优化点

- 新增 `--enable_multimodal_encoder_data_parallel`：开启后 Qwen2.5-VL vision tower 的所有并行线性层（MLP 的 gate_up/down、attention 的 qkv/proj、patch merger 的 fc）从 Column/Row-Parallel 改为 Replicated/MergedReplicated，attention 的 `tp_size` 视为 1，即每卡持有完整 ViT 权重。
- 新增 `run_dp_sharded_mrope_vision_model`：把当前 step 的多张图按 patch 总量贪心分配到 TP 组内各 rank（`get_load_balance_assignment`，按总尺寸而非图片数均衡），各 rank 独立跑完整 ViT，输出按 `max_len_per_rank` padding 后 `tensor_model_parallel_all_gather`，再去除 padding 并按原图顺序重组；处理空输入与不均衡负载。
- `_process_image_input`/`_process_video_input` 在 DP 开启时改走该分片函数；`linear.py` 的 `MergedColumnParallelLinear.weight_loader` 改为 `param.data[...] = loaded_weight`（配合 Replicated 权重加载）。
- 与既有非 mrope 的 `run_dp_sharded_vision_model` 并存，本 PR 是其 mrope（Qwen2.5-VL）变体。

## 关键改动

```text
vllm/multimodal/utils.py: get_load_balance_assignment（贪心按 patch 总量均衡）+ run_dp_sharded_mrope_vision_model
vllm/model_executor/models/qwen2_5_vl.py: vision tower 各子层按 use_data_parallel 选择 Replicated 线性层
vllm/model_executor/layers/linear.py: MergedColumnParallelLinear.weight_loader 的 param.data 赋值修正
tests/multimodal/test_utils.py: 负载均衡分配与 mrope DP 分片的多卡单测
```

## 性能声明

（Qwen/Qwen2.5-VL-72B-Instruct，ROCm（`VLLM_ROCM_USE_AITER=1`，8 卡 HIP），TP=8，ViT TP vs ViT DP 对照；benchmark_serving，VisionArena-Chat 1000 prompts，C=64。）

- 吞吐：Request 1.79→2.63 req/s（+47%）；Output token 206.66→302.89 tok/s（+47%）；Total token 375.53→551.31 tok/s（+47%）。
- 延迟：Duration 558.57→379.70 s（−32%）；Mean TTFT 4,584.50→2,062.35 ms（−55%）；P99 TTFT 25,097.62→10,313.05 ms（−59%）；Mean TPOT 285.83→198.03 ms（−31%）；P99 ITL 3,699.28→1,291.46 ms（−65%）。
- 反向点：Median ITL 33.11→131.26 ms（+296%，更慢），PR 未解释该退化。
- 作者结论：大部分改进来自 DP 化 Conv3d。

Accuracy（lm_eval ChartQA，Qwen2.5VL-72B）：TP8 baseline explicit/anywhere relaxed correctness 0.8864/0.8908；DP8+TP8 0.8844/0.8884，轻微下降。

## 验证与回归风险

- 新增多卡单测：`get_load_balance_assignment` 参数化用例（空输入/少于 GPU 数/单卡/均衡/不均衡）、`run_dp_sharded_mrope_vision_model` 与直接前向的 allclose 对比（bs 1/3/5）、空输入、4 卡不均衡负载、不同 spatial_merge_size。
- 回归风险：DP 开启时每卡持有完整 ViT 权重，显存占用上升（PR 未给数字）；贪心分配与 padding/all-gather 重组逻辑在图片尺寸极不均时行为需依赖单测；仅覆盖 Qwen2.5-VL 路径（qwen2_vl.py 仅清理注释）；Median ITL 退化未分析，可能反映 DP 分片后单卡 batch 变大。

## 关联

- 前序：非 mrope 版 `run_dp_sharded_vision_model`（本 PR 之前已存在于 `vllm/multimodal/utils.py`）。
- 同区域：本批多模态 vision 优化 PR #61458（PaddleOCR-VL encoder/projector 批量化）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
