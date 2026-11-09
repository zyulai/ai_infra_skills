# PR #28527: [Diffusion][CPU] Adding AMX optimizations for CPU platform

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jianan-gu |
| 合并 | 2026-07-09 |
| 分类 | Other |
| Label | Multi-modal, sgl-kernel, intel, cpu, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/28527 |

## 一句话结论

为 CPU 扩散模型引入 Intel AMX 三件套(AMX attention 后端、AMX weight-packed linear、VAE channels_last_3d),Wan2.2-TI2V-5B 各阶段提速 1.57x-10x。

## 动机

multimodal_gen 扩散管线在 CPU 平台上此前只有 Torch SDPA attention 与朴素 `F.linear`,没有利用 Intel AMX 矩阵扩展。本 PR 取 #20816 列出的部分 follow-up,把原本面向 LLM 模型的 AMX 优化搬到 CPU 扩散场景:替换 SDPA 的 attention 后端、AMX 权重打包 linear、VAE 的 channels last 3d 内存布局。

## 优化点

三条机制:(1) 新增 `AMXAttentionBackend`/`AMXATTNImpl`(改编自 FastVideo),forward 直接调 `sgl_kernel.flash_attn_varlen_func`;CPU 平台的 `get_attn_backend_cls_str` 在检测到 AMX 支持时自动选 AMX_ATTN,否则回退 SDPA。(2) `UnquantizedLinearMethod` 增加 `process_weights_after_loading` 调 `_amx_process_weight_after_loading` 做 AMX 权重重排,`apply` 在 `use_intel_amx_backend(layer)` 成立时改走 `torch.ops.sgl_kernel.weight_packed_linear(is_vnni=True)`;text_encoder_loader 与 fsdp_load 在加载后对带 `quant_method` 的模块统一执行该处理,cpu offload 场景用 `device_loading_context` 临时搬回设备再处理。(3) VAE 的 channels_last_3d 支持从 CUDA/ROCm 扩展到 CPU(`vae_loader._should_use_channels_last_3d` 与 wanvae 平台检查)。配套改动:sgl-kernel CPU `flash_attn_varlen_func` 新增可选 `sm_scale` 参数(原先硬编码 `1/sqrt(head_size)`),torch extension schema 改为 `float? sm_scale`,`vision.py` 把 softmax_scale 透传下去。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/amx_attn.py: 新增 AMX attention 后端(调 sgl_kernel flash_attn_varlen_func)
python/sglang/multimodal_gen/runtime/layers/linear.py: AMX 权重打包 + weight_packed_linear 分派
python/sglang/multimodal_gen/runtime/platforms/cpu.py: CPU 自动选择 AMX_ATTN 后端
python/sglang/multimodal_gen/runtime/loader/component_loaders/vae_loader.py: channels_last_3d 扩展到 CPU
sgl-kernel/csrc/cpu/flash_attn.cpp: flash_attn_varlen_func 支持自定义 sm_scale
```

## 性能声明

以 Wan-AI/Wan2.2-TI2V-5B-Diffusers 为例(PR 正文,未注明硬件型号):TextEncodingStage 提速 10x(bf16 encoding;fp32 不变);DenoisingStage 提速 2x;DecodingStage 提速 1.57x(bf16、开启 channel last)。正文还指出后续可通过 rmsnorm/layernorm 与新 GELU 的 fusion(compile 或 kernel 级)再获得小幅提升,留待 follow-up。

## 验证与回归风险

既有 `test/registered/cpu/test_flash_attn.py` 扩展覆盖 `softmax_scale=[None, 0.2]`(varlen、large size 与 causal mask 用例),回归保护 sm_scale 改动;PR 未提供端到端精度评测结果。风险面:AMX 探测在模块导入时执行(`cpu_has_amx_support()`);所有 CPU 且有 AMX 的 diffusion 模型 linear 层都切换到 weight_packed_linear 路径;channels_last_3d 在 CPU VAE 上默认启用(可用环境变量覆盖);`flash_attn_varlen_func` schema 变更影响其他 CPU 调用方(默认 None 保持旧行为)。PR CI:Base 与 Extra 均失败(x)。

## 关联

前序:PR #20816(follow-up 清单来源)。同批 diffusion:PR #35698([diffusion] LTX-2.5 decoder 3D RoPE,GPU 侧)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
