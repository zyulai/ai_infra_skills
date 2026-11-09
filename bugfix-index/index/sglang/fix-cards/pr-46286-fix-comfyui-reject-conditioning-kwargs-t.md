# PR #46286: [Fix] comfyui: Reject conditioning kwargs the SGLD adapters silently drop

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46286](https://github.com/sgl-project/sglang/pull/46286) |
| 作者 | OnePunchMonk |
| 合并日期 | 2026-10-28 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ComfyUI 适配器 `pack()` 把不支持的 conditioning kwargs 静默丢弃、发出错误请求;改为显式抛 `ValueError`。

## 问题现象
ComfyUI 原生 `QwenImageTransformer2DModel.forward` 会传 `attention_mask`、`additional_t_cond` 以及经 `**kwargs` 传入的 `ref_latents_method`,`QwenImageEditAdapter.pack()`(`apps/ComfyUI_SGLDiffusion/executors/qwen_image.py`)把三者全部收进 `**kwargs` 且从不转发:无论是否设置,发给 SGLD worker 的请求完全相同,使用它们的工作流静默按 worker 硬编码默认值运行(worker 的 `QwenImagePipeline` 在加载时按 checkpoint 固定 `zero_cond_t`,reference latents 总按原生 ComfyUI 默认 `ref_latents_method="index"` 的方式应用)。`ZImageAdapter.pack()`(Lumina2 / Z-Image)对 Z-Image Omni 的 `siglip_feats` / `ref_contexts`(多参考图 conditioning)有同样缺口:worker 的 `ZImagePipeline` 无对应路径,带参考图的工作流会静默产出纯文生图结果。

## 根因
正文披露:两个 `pack()` 用 `**kwargs` 捕获不支持的参数且从不转发,worker 端没有对应实现路径,导致静默错误结果而非报错。与 #43404 修复 `FluxAdapter` 的 `ref_latents`/`control` 是同一失效模式。

## 修复方案
两个 `pack()` 由静默丢弃改为显式拒绝:
- `QwenImageEditAdapter`:非 `None` 的 `attention_mask` / `additional_t_cond` 抛 `ValueError`;`ref_latents_method` 非默认值(`!= "index"`)抛 `ValueError`;
- `ZImageAdapter`:非空的 `siglip_feats` / `ref_contexts` 抛 `ValueError`。

拒绝范围经核对原生 ComfyUI 源码(`comfy/ldm/qwen_image/model.py`、`comfy/ldm/lumina/model.py`、`comfy/model_base.py`)确定:`ref_latents_method` 默认 `"index"` 且 worker 只支持该模式,故仅拒绝非默认值;Qwen-Image 的 `attention_mask`/`additional_t_cond` 在 `QwenImage.extra_conds` 中仅在有意义时设置,拒绝非 `None` 不会误伤普通工作流;Z-Image 故意**不**拒绝 `num_tokens`/`attention_mask`(`Lumina2.extra_conds` 每次调用都发 `num_tokens`,原生 `patchify_and_embed` 本就丢弃 `attention_mask` 的效果,丢弃已与原生行为一致)。并扩展 `test_comfyui_adapters.py` 覆盖拒绝与放行两类用例。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `QwenImageEditAdapter.pack()` 与 `ZImageAdapter.pack()` 的参数校验;此前依赖静默丢弃的工作流现在会收到明确报错(有意的行为改变)。纯 Python kwarg 校验、无模型权重与 worker 往返;作者在 Modal CPU 上以 stub 方式对修改源文件直接验证,9/9 通过,含默认参数行为不受影响的回归检查。

## 关联
#43404

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
