# PR #38040: [Fix] Misc Fixes in ViT CUDA Graph

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @b-mu |
| 合并 | 2026-05-14 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | performance, ready, v1, multi-modality, qwen, nvidia |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/38040 |

## 一句话结论

修复 ViT encoder CUDA Graph 的 budget 推导与分配缺陷,使 Qwen3-VL 全图 capture 可用,encoder 前向 P99 最多快 42.6%。

## 动机

ViT CUDA Graph 路径存在三个问题:(1) 自动推导时 `max_batch_size = max_budget // min_budget` 可能超过 `min_budget`,导致 `per_image_output = token_budget // max_batch_size = 0`,在 `Qwen3_VisionPatchEmbed.forward` 对空 tensor reshape 时崩溃——小 budget 下 capture 直接失败;(2) `token_budget` 不能整除 `max_batch_size` 时 floor 除法会少分配输入 buffer;(3) Qwen3-VL 上此前只对 video 禁用 CUDA graph(EVS pruning 时),但 `embed_multimodal` 对 image embedding 也有后处理(mrope positions 追加),graph 路径绕过该后处理会产生与 eager 不一致的 embedding 格式。

## 优化点

三处修复:(1) 在 `EncoderCudaGraphManager.__init__` 建立并强制不变式 `max_batch_size <= min(token_budgets)`:全自动推导时取 `min(max_budget // min_budget, min_budget)`;用户只给 max_vision_items 时上调 budget 下界;只给 budgets 时 cap max_batch_size;两者都给但不满足不变式则 raise,并校验 budget 范围(正数、min<=max),`CompilationConfig` 也拒绝非正 budget;(2) `per_mm_item_output` 改为 ceil 除法,保证最坏情况(单个 item 用满 budget)下 buffer 够用;(3) `qwen3_vl.get_encoder_cudagraph_config` 在 pruning 开启时返回空 modalities(image、video 都不 capture),pruning 关闭时恢复 image+video。

## 关键改动

```text
vllm/v1/worker/encoder_cudagraph.py: max_batch_size/budgets 四种配置路径的不变式校验
vllm/model_executor/models/qwen3_vl.py: per_mm_item_output 改 ceil;pruning 开启时禁用全部 encoder CUDA graph
vllm/config/compilation.py: encoder_cudagraph_token_budgets 非正值校验
tests/v1/cudagraph/test_encoder_cudagraph.py: 新增 TestInitInvariantValidation(无需 GPU)
```

关键片段(encoder_cudagraph.py):

```python
# Invariant: max_batch_size <= min_token_budget.
# This ensures per_image_output = budget // max_batch_size >= 1
# for every captured budget, preventing reshape crashes on empty
# tensors during CUDA graph capture.
```

## 性能声明

测试条件:4x GB200 NVLink(TP=4、ViT DP=4)、Qwen3-VL-32B-Instruct、random-mm 合成数据集(1000 prompts、200 warmups、max-model-len 16384、bfloat16、FLASH_ATTN、mm-encoder-tp-mode data),对比 Eager 与 ViT full CUDA Graph(`cudagraph_mm_encoder: true, encoder_cudagraph_max_frames_per_batch: 1`)。Encoder Forward Latency:mean FA4 BF16 38.4 → 29.7 ms(+22.7%)、FLASHINFER FP8 39.5 → 31.5 ms(+20.2%);median 25.4 → 23.9 ms(+5.8%)、29.8 → 25.3 ms(+15.1%);P99 194.0 → 155.6 ms(+19.8%)、244.4 → 140.3 ms(+42.6%)。

## 验证与回归风险

新增 `TestInitInvariantValidation` 覆盖全部配置路径(自动推导、坏组合 raise、边界值、非正 budget),无需 GPU。风险面:pruning 开启时 Qwen3-VL encoder 不再走 CUDA graph,该配置下失去相应加速;ceil 分配略增大 buffer 占用;不变式校验会对先前"能跑但错"的用户配置报错。PR 未注明其他新增测试。

## 关联

同分类 CUDA Graph / Compiler:PR #29207(主模型 graph capture stream);区域为 ViT encoder cudagraph(`encoder_cudagraph_*` 配置族)。PR 未注明前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
