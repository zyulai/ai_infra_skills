# PR #56277: [MM] Mistral vision Encoder CG support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NickLucche |
| 合并 | 2026-11-03 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, performance, ready, multi-modality, nvidia, mistral |
| Diff | diffstat 缺失(pack 未记录);diff 片段覆盖 4 个文件,另有 4 个文件因篇幅从 pack 中省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/56277 |

## 一句话结论

为 Mistral 系多模态模型的 Pixtral vision encoder 启用 encoder CUDA Graph,小模型 encoder 延迟降至约 1/4.6。

## 动机

此前 Mistral 多模态模型(Pixtral 与 Mistral3 两种架构)的视觉 encoder 只能 eager 执行:变分辨率图像导致输入 shape 不固定,无法用 CUDA Graph 捕获。encoder forward 在小模型上占比高(Ministral-3-3B 单次 14.32 ms),是端到端延迟的明显瓶颈。PR #54214 已独立实现原生 Pixtral 路径,本 PR 在其基础上扩展并补齐 Mistral3 路径。

## 优化点

核心机制是把变分辨率的图像 patch 打包进固定容量的 graph 缓冲区:所有图像的 patch 被 `pack_image_patches` 打包成单条 patch 序列,再用零长度序列补齐到 `max_batch_size`,使 cu_seqlens 等 metadata shape 在 capture 与 replay 间保持一致,attention 对 padding 行不做实际计算。attention metadata、2D rotary embedding 的索引、patch-merger 的 gather 索引均预计算成固定 shape buffer,replay 时只需拷贝新值。`max_seqlen` 特意保留在 CPU 上,因为 attention 后端在 capture 期间会对它调用 `.item()`,留在 CPU 可避免 graph 内记录 D2H 拷贝与 device sync。不满足条件的配置(SDPA 后端、动态 FP8 scale、预计算 embedding 输入)自动回退 eager 执行。

## 关键改动

```text
vllm/model_executor/models/pixtral.py: 新增 PixtralEncoderCudaGraphMixin(捕获/回放
  buffer 定义、graph 安全性判定)及 pack_image_patches、零长度 padding metadata
vllm/model_executor/models/mistral3.py: Mistral3ForConditionalGeneration 混入该
  Mixin,projector 增加 forward_packed 路径
tests/models/multimodal/generation/test_vit_cudagraph.py: 增加 pixtral/mistral3
  两个 encoder CG 端到端测试配置
tests/models/multimodal/generation/test_pixtral.py: packed metadata/merge 索引单元测试
```

关键机制摘录(pixtral.py,零长度序列 padding 保 shape 固定):

```python
lengths = np.array(sequence_lengths, dtype=np.int32)
if num_sequences is not None:
    lengths = np.pad(lengths, (0, num_sequences - len(lengths)))
```

## 性能声明

mistralai/Ministral-3-3B-Instruct-2512,单卡 H200,TP=1,224×224 图像,FlashAttention,200 条测量请求(`vllm bench mm-processor`,FULL_DECODE_ONLY graph 模式):

| 指标 | CG off | CG on | 加速 | 降幅 |
| --- | --- | --- | --- | --- |
| Encoder mean | 14.32 ms | 3.10 ms | 4.62× | 78.4% |
| Encoder median | 13.25 ms | 3.07 ms | 4.32× | 76.8% |
| Encoder P90 | 16.07 ms | 3.18 ms | 5.05× | 80.2% |
| Encoder P99 | 24.78 ms | 3.30 ms | 7.51× | 86.7% |
| E2E mean | 2058.45 ms | 991.87 ms | 2.08× | 51.8% |

作者注明小模型上 encoder 收益最大;另在 Mistral Large 3 NVFP4 TP=4 上做了实际 serving 验证(未附数字)。

## 验证与回归风险

新增/依赖测试:`test_pixtral.py` 的零长度 padding、FP8 layout、merge indices 对齐 unfold、pack 与 conv 一致性等单元测试;`test_vit_cudagraph.py` 覆盖 Pixtral 与 Mistral3 两条路径的端到端生成。覆盖 FlashAttention、FlashInfer、Triton、ROCm AITER 的 metadata 路径。风险点:依赖 processor 把图像 resize 到 merged patch size 的整数倍(grid 不整除时直接抛错);SDPA 后端与动态 FP8 scale 场景仍走 eager;`max_seqlen` 以 capture 上限固化进 graph,replay 不更新。

## 关联

- PR #54214:独立的原生 Pixtral encoder CG 实现,与本 PR 部分重叠(本 PR 声明为其添加 co-author)
- 同区域:多模态 encoder CUDA Graph 框架(`encoder_cudagraph_defs`、`SupportsEncoderCudaGraph` 接口)及 `test_vit_cudagraph.py` 中已有的 gemma3 等模型配置

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
