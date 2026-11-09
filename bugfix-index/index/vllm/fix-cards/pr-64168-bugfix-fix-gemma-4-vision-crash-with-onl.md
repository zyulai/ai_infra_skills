# PR #64168: [Bugfix] Fix Gemma 4 vision crash with online FP8 quantization

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64168](https://github.com/vllm-project/vllm/pull/64168) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-11-06 |
| labels | bug, ci/build, multi-modality, nvidia, quantization, ready |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
online FP8 下 Gemma 4 图像请求 500 并杀死引擎;patch embedder 改以 model_dtype 执行,量化层保持 FP8。

## 问题现象
`--quantization fp8` 下对 Gemma 4 发图像请求返回 HTTP 500 并使 engine core 死亡,日志为 `RuntimeError: ... "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type" not implemented for 'Float8_e4m3fn'` 与 `EngineDeadError`(3/3 次复现;修复后 6/6 返回 200)。视频在 `--enforce-eager` 下引擎启动即崩(encoder profile run 发送 dummy video);online `fp8_per_block`、`fp8_per_channel` 与 `mxfp8`(eager、custom ops off)同样启动即崩;借用 Gemma 4 `_process_image_input`/`_process_video_input` 并自建 FP8 tower 的 `DiffusionGemmaForConditionalGeneration` 与 `EmbeddingGemma2Model` 也崩溃。

## 根因
正文披露:vLLM 用 `AutoModel.from_config` 构建 Gemma 4 vision tower,再把每个 `nn.Linear` 换成携带 tower quant config 的 `ReplicatedLinear`;online FP8 下 `input_proj.weight` 载入后为 `float8_e4m3fn`。transformers `Gemma4VisionPatchEmbedder.forward` 把 pixel_values cast 到 `input_proj.weight.dtype`——对量化层这是存储 dtype;FP8 属浮点类型所以 guard 放行,bf16 pixel values 被无 scale 舍成 FP8,FP8 linear 的 activation quant 收到 FP8 输入,而 `dynamic_per_token_scaled_fp8_quant` 只接受 fp16/bf16/fp32。`encoder_cudagraph_forward` 在 vLLM 代码里犯同样的错:把 pooled states cast 到 `embedding_projection.weight.dtype`(同为 FP8)。

## 修复方案
`vllm/model_executor/models/gemma4_mm.py` 新增模块级 `_embed_patches(...)`:复刻 transformers forward 步骤(`2*(x-0.5)`、`input_proj`、`_position_embeddings`、求和)但按传入 dtype cast;图像、视频与 encoder CUDA graph 路径以 `self.model_dtype` 调用:
```python
embedder = self.vision_tower.patch_embedder
pixel_values = (2 * (pixel_values - 0.5)).to(self.model_dtype)
hidden_states = embedder.input_proj(pixel_values)
position_embeddings = embedder._position_embeddings(
    pixel_position_ids, padding_positions)
return (hidden_states + position_embeddings).to(self.model_dtype)
```
CUDA graph 路径 pooled states 亦改 cast 到 `self.model_dtype`(替换原 `embedding_projection.weight.dtype` cast)。quant config 不动,层保持 FP8;未量化层 weight dtype 即 model dtype,无变化。helper 用模块级函数而非方法,因两个借用模型复制方法而不继承 Gemma 4。新增测试在旧代码上对 image/video 与 vit_cudagraph 各 2 failed,本 PR 后 4 passed。

## 影响与回归风险
- 影响窗口:验证于 vLLM `00b6c4c56`(0.32.1rc1.dev505)、transformers 5.20.2;正文未给出受影响版本范围。
- 回归面:默认 compile 模式(此前不崩)图像输出可能微变——pixel values 不再先舍到 FP8,input_proj 相对误差 4.50%→3.68%,未跑 vision eval;`_embed_patches` 依赖 transformers 私有 `_position_embeddings`;AMD 未验证;online mxfp4 在 SM 12.x 的启动失败与本 patch 无关(前后相同)。

## 关联
Fixes #63732;对比 PR #63911

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
