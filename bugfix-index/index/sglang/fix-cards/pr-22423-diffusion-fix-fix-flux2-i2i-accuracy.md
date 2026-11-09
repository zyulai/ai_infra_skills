# PR #22423: [diffusion] fix: fix flux2 i2i accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22423](https://github.com/sgl-project/sglang/pull/22423) |
| 作者 | mickqian |
| 合并日期 | 2026-04-10 |
| labels | documentation, npu, run-ci, diffusion |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
FLUX.2 i2i 输出与 diffusers 不一致;对齐 Mistral3 encoder、VAE 编解码 BN 归一化、条件图预处理与 SDPA 后端等行为修复精度。

## 问题现象
FLUX.2-dev 在 sglang 下与 diffusers 输出不一致。正文给出复现命令(diffusers vs sglang 对比图):
```
sglang generate \
    --model-path black-forest-labs/FLUX.2-dev \
    --save-output \
    --prompt "Convert 2D style to 3D style" \
    --image-path https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg \
    --seed 42
```
以及一个纯文生图命令(`--prompt "Doraemon is eating dorayaki" --width 1024 --height 1024 --seed 42`),均附对比截图,显示 sglang 生成的图像与 diffusers 基线存在可见差异。正文未给量化指标。

## 根因
正文未给出单一根因,而是以 Modifications 列出三类与参考实现不对齐的行为:1) consistency check 失败时未打印 GT 链接;2) mistral encoder 行为未对齐;3) vae decode 行为未对齐。diff 中的注释进一步披露(由 diff 推断,引自 diff):「FLUX.2 uses the text-only Mistral3 path but still expects the same local SDPA kernel choice as the official HF implementation」「FLUX.2 stays closer to the official diffusers output with Torch SDPA. The generic FA path still produces a measurable image-level drift here」;Flux2 VAE latent 需按 VAE 的 BN running mean/var 归一化、条件图需按 1024*1024 目标面积而非直接 resize 到目标宽高处理。

## 修复方案
(注:diff 在 40000 字符处截断,以下基于已获取部分;`test_server_common.py` 之后可能还有未展示文件。)
- `configs/pipeline_configs/base.py` 新增 `normalize_vae_encode` 钩子;`flux.py` 的 `Flux2PipelineConfig` 实现:VAE 含 `bn` 时用 `running_mean`/`running_var+batch_norm_eps` 对 image latents 归一化,并新增 `vae_precision: str = "bf16"`;`preprocess_condition_image` 改为超面积时按 `1024*1024` 目标面积等比缩放(优先调用 diffusers 的 `_resize_to_target_area`)。
- `pipelines/flux_2.py`:图像处理器由 `VaeImageProcessor` 换成 `Flux2ImageProcessor`;`stages/image_encoding.py` 走 `normalize_vae_encode` 分支替代通用 scale/shift。
- `models/encoders/mistral_3.py`:对齐 HF Mistral3 实现——按 `config._attn_implementation` 选 attention interface、支持 sliding window causal mask、`MistralModel` 继承 `MistralPreTrainedModel` 并调用 `post_init`、CUDA 下用 `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)` 上下文、移除自定义 `USPAttention` 包装。
- `models/dits/flux_2.py`:为 `Flux2Transformer2DModel` 限定 `_supported_attention_backends = {TORCH_SDPA, FA}` 并向下传递。
- `component_loader.py`:按 `model_index.json` 声明的组件类(如 `PixtralProcessor`)而非槽位名选择 AutoTokenizer/AutoProcessor;`text_encoding.py` 输出统一搬到 `target_device`。
- `input_validation.py` + `sampling_params.py`:记录用户显式指定的 `width`/`height`(`_explicit_fields`),仅显式指定时才覆盖按条件图计算出的尺寸。
- `test/server/test_server_common.py`:一致性检查失败时对图像与视频统一打印 GT 文件链接(原只对 video 打印)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`multimodal_gen` 中 Flux2 相关路径(配置、pipeline、DiT、Mistral3 文本编码器、图像编码/输入校验阶段);`for_component_type` 新增 `component_architecture` 参数影响所有组件加载调用方;`normalize_vae_encode` 默认返回 None,其余 pipeline 走原 scale/shift 路径;显式 width/height 语义变化影响传参行为;diff 截断,可能存在未展示的其他改动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
