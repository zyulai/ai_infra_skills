# PR #44865: [Feat] ComfyUI: Add online INT8 quantization for BF16 Qwen-Image-2.1 single files

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @OnePunchMonk |
| 合并 | 2026-10-21 |
| 分类 | Quantization (tier: major) |
| Label | documentation, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/44865 |

## 一句话结论

让 ComfyUI 的 BF16 Qwen-Image-2.1 单文件支持在线 INT8 量化,采样约 1.4x 提速、DiT 权重显存减半。

## 动机

24-32 GB 显卡的用户往往只有 BF16 的 ComfyUI 单文件,想做在线量化却必须另找预量化文件。两个原因堵死了这条路:插件 preflight 对所有 ComfyUI 单文件拒绝 `quantization` 选项;worker 的单文件加载器 `comfyui_checkpoints/spec.py` 只从文件内 per-layer marker 构建 quant config,BF16 文件没有 marker,于是 `--quantization` 被静默忽略,DiT 停留在 BF16。

## 优化点

改动在加载路径而非 kernel。当单文件没有 quant marker 且 `quantization` 已设置时,`load_comfyui_transformer` 走与 native 目录相同的 `resolve_transformer_quant_load_spec`,得到 quant config、param_dtype 与 post-load hooks,并构建对应 `WeightLoadPlan`(ConvRot INT8 允许从 host 权重起量化,checkpoint_load_device 为 CPU;FP8 等从 GPU 量化)。加载完成后逐个执行 post_load hook 完成 online 量化。preflight 侧新增白名单 `_SINGLE_FILE_ONLINE_QUANT = {"qwen_image21"}` 与 `_SINGLE_FILE_ONLINE_METHODS = {"convrot_int8"}`,其余模型/方法仍拒绝并给出明确报错;需要 serialized 权重的方法(如 `mxfp8`)raise 而不是被忽略。FP8 实测有漂移且更慢,故未放开。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/loader/comfyui_checkpoints/spec.py: BF16 单文件无 marker 时的在线量化解析、WeightLoadPlan 与 post-load hooks
python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/core/preflight.py: 放行 qwen_image21 + convrot_int8,其余组合明确拒绝
python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/options.py: quantization tooltip 更新
test_comfyui_checkpoint_spec.py / test_comfyui_options.py: 新增 worker 回归与 preflight accept/reject 用例
```

## 性能声明

Modal L40S、ComfyUI v0.41.0 headless、Comfy-Org/Qwen-Image-2.1 turbo 文件、1024x1024、8 steps、euler/simple、cfg 1、seed 42,取第二次(热)运行,峰值显存为整进程 nvidia-smi:

- SGLD BF16:3.29 s / 26.9 GB(reference)
- SGLD online INT8(本 PR):2.35 s / 23.5 GB,约 1.4x,PSNR 28.55 dB / SSIM 0.966(vs SGLD BF16)
- worker 日志:quantized 224 linear layers,13.00 GiB BF16 -> 6.50 GiB INT8
- online FP8 未启用:4.36-4.49 s,SSIM 0.855 左右,比 BF16 慢且漂移(L40S CC 8.9 无 JIT INT8 kernel,走 comfy_kitchen backend)

## 验证与回归风险

新增 worker 回归测试(quant config、param dtype、load device、hooks 到达 loader)与 preflight accept/reject 用例;Modal 上单测 64 passed、3 skipped,同批测试在未改的 main 上 5 failed,确认测试有守卫作用。风险面:仅覆盖 Qwen-Image-2.1 + INT8 ConvRot,Flux/Qwen-Image/Z-Image/MiniMax-H3 维持拒绝;FSDP 推理 + 在线 ConvRot INT8 未测试;本 PR 不触碰 serialized-marker 路径(PR 指出预量化 `qwen_image_2.1_turbo_int8_convrot.safetensors` 在 main 上加载仍报错)。

## 关联

同属 ComfyUI/diffusion 量化路径:与 native 目录在线量化共用 `resolve_transformer_quant_load_spec` 与 `WeightLoadPlan`;PR 提及 serialized-marker 预量化文件加载 bug 拟另开 issue。pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
