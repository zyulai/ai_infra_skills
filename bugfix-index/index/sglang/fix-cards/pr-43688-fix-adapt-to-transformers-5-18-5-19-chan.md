# PR #43688: [Fix] Adapt to transformers 5.18 / 5.19 changes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#43688](https://github.com/sgl-project/sglang/pull/43688) |
| 作者 | mmangkad |
| 合并日期 | 2026-10-12 |
| labels | documentation, run-ci, diffusion, release-highlight |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论
跟进 transformers 5.18/5.19 上游变更:GGUF tokenizer 改用 tokenizers 直建、接受 renamed 的 indexed_attention、移除失效的 backport 与已删 API。

## 问题现象
在 transformers 5.19 pin 之上,5.18/5.19 的上游变更使 SGLang 多处失效:DeepSeek GGUF tokenizer 依赖的 converter 在上游被移动,且新 converter 会偏移 special-token id;SGLang 自带的嵌套 `rope_parameters` 校验 backport 已被上游收编而冗余;DSA layer 类型在上游更名为 `indexed_attention`,导致 HY-V4 / Qwen4-Exp 配置校验不认;`_commit_hash` 属性与 NPU MiniMax-M3 的 `do_convert_rgb` 参数被移除,读取即报错。

## 根因
正文披露:上述问题均为 transformers 5.18 / 5.19 的上游 API/行为变更所致( converter 移动、layer type 更名、属性与参数删除、校验逻辑上游化),SGLang 侧代码未同步适配。

## 修复方案
按正文 Summary 与 diff 逐项适配:
- `python/sglang/srt/model_loader/expert_pack_runtime.py`:DeepSeek GGUF tokenizer 改为直接用 `tokenizers` 构建 BPE(`models.BPE` + `decoders.ByteLevel` + `processors.ByteLevel(trim_offsets=False)`),不再经 `transformers.integrations.ggml.convert_gguf_tokenizer`,避免 special-token id 偏移。
- `python/sglang/srt/utils/hf_transformers_patches.py`:删除 `_patch_nested_rope_validation` backport(上游 huggingface/transformers#48798 已收编,适用条件 transformers >= 5.18)。
- `python/sglang/srt/configs/hy_v4.py`、`qwen4_exp.py`:接受更名后的 `indexed_attention` layer 类型(HY-V4 校验白名单加入,Qwen4-Exp 的 `layers_block_type` 将其映射为 attention)。
- `python/sglang/srt/arg_groups/model_hook.py`:改用 `model_config.revision`,不再读取已删除的 `hf_config._commit_hash`。
- `python/sglang/srt/hardware_backend/npu/modules/minimax_m3_processor.py`:删除 NPU 视频预处理包装器中已移除的 `do_convert_rgb` 参数。
- 注释修正(Hunyuan3D 的 Dinov2 布局变更实际落在 5.18 而非 5.19;Granite 的 layer_types 在 config init 时被重写)与 GLM-5.3-Flash cookbook 文档更新。

## 影响与回归风险
- 影响窗口:正文未说明;针对 transformers 5.18/5.19(在 5.19 pin 之上)。
- 回归面:DeepSeek-V4-Flash MXFP4 GGUF(`expert_pack`)的 tokenizer 生成路径——正文验证 tokenizer 文件与之前 byte-identical、输出一致;HY-V4 / Qwen4-Exp 配置加载与校验;granitemoehybrid/hunyuan3d 相关注释与 remap 逻辑;NPU MiniMax-M3 视频预处理;model_hook 的权重 dtype 探测路径。作者验证:B300 上 cookbook flags,5.18/5.19 vs 5.17 的 GSM8K(Qwen3.8-Flash-Next 97.95、Hy4-preview-FP8 96.58、MiniMax-M2 95.89、GLM-5.3-Flash 97.34,图像/视频 OK);5.19 vs 5.18 在 51 个 checkpoint 上 config/tokenizer/图像视频处理器输出一致。CI 状态:Base 通过,Extra 与 AMD ROCm 10 失败。

## 关联
- https://github.com/huggingface/transformers/pull/48798 (被删 backport 对应的上游修复,正文 diff 注释提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
