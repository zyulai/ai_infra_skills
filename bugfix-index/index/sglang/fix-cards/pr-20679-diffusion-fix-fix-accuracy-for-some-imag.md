# PR #20679: [diffusion] fix: fix accuracy for some image models

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20679](https://github.com/sgl-project/sglang/pull/20679) |
| 作者 | mickqian |
| 合并日期 | 2026-03-22 |
| labels | npu, DO NOT MERGE, run-ci, diffusion |
| 热度 | 👍 1 · 💬 12 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
diffusion 多管线与官方 diffusers 行为不一致(CFG、负向分支、RoPE 精度、SP 分片等)导致 Qwen-Image/Edit、Z-Image 出图失真;按 diffusers 语义逐项对齐修复。

## 问题现象
若干图像生成模型在 sglang `multimodal_gen` 下输出与官方 diffusers 明显不一致:Qwen-Image 文字渲染错误且构图失真(修复前/修复后/diffusers 三方对比图见原帖);Qwen-Image-Edit 编辑结果偏离 diffusers(附 abs_diff 可视化);Z-Image-Turbo 出图与 diffusers 不一致。以上在 `--ulysses-degree 2` 的 SP 路径下同样或更严重。正文给出可复现命令(`sglang generate`,Qwen-Image 1664×928、50 步、true_cfg_scale 4.0、seed 42;Z-Image-Turbo 1024×1024、9 步、guidance_scale 0.0、seed 42)及对应 diffusers 参考脚本。

## 根因
正文在 Modifications 中逐项披露了与官方实现的偏差(diffusers 为参考基准):
1. `Qwen-Image` 未实现官方的 true-CFG:应引入 `true_cfg_scale`、大于 1 才启用 CFG,并匹配官方 true-CFG 的 norm rescale。
2. `Qwen-Image-Edit` 未尊重 negative-image,uncond 分支实际没有用 `negative_prompt`。
3. `Qwen-Image-Edit` 的 Qwen2.5-VL vision rotary 频率未保持 fp32(应与 diffusers 对齐以减少 encoder 漂移)。
4. `Qwen-Image-Edit` 在 `ulysses-degree=2` 下未对 condition-image 的 rope 与 noisy-image 的 rope 一并分片,`zero_cond_t` modulation 索引未按本地 SP 序列长度构建。
5. `Z-Image` 的 prompt/tokenization 与 dtype 错误:应使用渲染后的 chat-template prompt、bf16 文本编码、fp32 的 latent 采样/scheduler 状态。
6. `Z-Image` SP/Ulysses 下错误地分片了 caption tokens(condition 不应分片),应作为 replicated suffix 参与联合注意力,且 local/global RoPE 偏移错误导致多 GPU 与单 GPU 不一致。

## 修复方案
按上述六项,改动集中在 `python/sglang/multimodal_gen/` 下:
1. `configs/pipeline_configs/base.py` 新增 CFG 钩子 `get_classifier_free_guidance_scale` 与 `postprocess_cfg_noise`、`get_latent_dtype`、`gather_noise_pred_for_sp`;`qwen_image.py` 实现 true-CFG(`true_cfg_scale > 1` 才启用,组合后按 per-token norm 回归 conditional 分支)并新增 `--true-cfg-scale` CLI 参数(`sampling_params.py`);修复 `prepare_image_processor_kwargs` 负向分支用逐 prompt 列表构造。
2. `qwen_image.py` 抽出 `_shard_qwen_edit_freqs_cis_for_sp`,同时分片 noisy 与 condition 部分的 rope cache(含 cos/sin 形态);diff 中另有 `zero_cond_t` 相关改动的截断部分。
3. vision rotary fp32 对齐在截断的 diff 部分,无法从本素材确认细节。
4. `zimage.py`:`tokenize_prompt` 改为先 `apply_chat_template(tokenize=False)` 渲染 prompt 再整体 `tokenizer(...)`;`text_encoder_precisions=("bf16",)` 使文本编码走 bf16(`text_encoder_loader.py` 新增 `load_native` 按精度用 `AutoModel.from_pretrained` 加载);`get_latent_dtype` 返回 fp32(latent 采样与 scheduler 状态用 fp32,对齐官方 pipeline)。
5. `zimage.py` 停止分片 caption tokens:删除 `_shard_cap` 与 plan 中的 cap 分片字段,caption 作为 replicated prefix 保留并修正 RoPE 偏移(offset 改为 `cap_ori_len + cap_padding_len + 1`);`runtime/layers/attention/layer.py` 的 `USPAttention` 新增 `_forward_with_replicated_suffix` 支持 replicated suffix(复用 replicated-prefix 路径加旋转),prefix+suffix 同时存在时报错;`flux.py`/`glm_image.py`/`qwen_image.py` 的联合注意力传入 `num_replicated_prefix`。
6. 其他:`layernorm.py` 的 `apply_qknorm` 为 fused in-place 路径增加 dtype 匹配守卫;`sana.py` 禁用 spatial latents 的序列分片;`selector.py` 单一支持后端时自动选中。

注:pack 的 diff 在 40000 字符处截断(文件粒度),第 2、3 项等涉及文件的完整改动无法从本素材确认。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:全部集中在 `multimodal_gen` 的 diffusion 管线:Qwen-Image/Edit、Z-Image 的采样数值与多 GPU(SP/Ulysses)路径变化最大;CFG 钩子对其他管线为默认透传(返回原 noise_pred/guidance_scale),`get_latent_dtype` 默认返回 prompt dtype,行为不变;`USPAttention` 新参数默认 0,既有调用路径不变;`apply_qknorm` 新守卫可能让原本走 fused kernel 的 fp32/fp16 混合精度路径回落普通实现(精度更稳);`sana.py` 的 SP 分片被显式关闭。labels 含 `DO NOT MERGE`,与 `npu` 标记并存,合并状态以仓库为准。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
