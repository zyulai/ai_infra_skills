# PR #29742: [diffusion] Fix Z-Image accuracy

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @qimcis |
| 合并 | 2026-07-08 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 jit kernel、zimage pipeline 配置、attention 层与 zimage DiT,另省略 7 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/29742 |

## 一句话结论

修复 Z-Image-Turbo 在动态 batching 下的图像质量劣化并贴近原生实现,顺带单/批生成均快约 2%。

## 动机

修复 #28502 及 Z-Image-Turbo 的普遍精度问题:`--batching-mode dynamic` 下请求被批处理后图像严重劣化。作者定位到两层原因:其一,qwen3 text encoder 的 `position_ids` 在 hidden_states 已是 `[batch, seq, dim]` 时仍是 `[1, seq]`,导致批处理 prompt 的 RoPE 按错误的扁平 token 布局应用;其二,当前实现相对原生 PyTorch 实现整体劣化(单张生成也一样)——采样与归一化不同(用了 scheduler 默认 sigma 路径、外层 autocast、共享的 fp32 累加 RMSNorm),且批处理把图像/caption pad 到统一长度时既不保留每个请求的 RoPE offset、也不在 attention 中 mask 批处理填充,混合 prompt 的动态批会 attend 到无效 token 并用错图像位置。

## 优化点

对齐原生实现的三个机制:新增 `ZImageRMSNorm`(全程 bf16,不上转 fp32)替换共享 RMSNorm,pipeline 配置 `enable_autocast=False` 关掉外层 autocast,`prepare_sigmas` 走原生 sigma 路径。批处理正确性:`_build_batched_freqs_cis` 逐样本构造并 pad 每个请求的 RoPE freqs(结果按形状缓存),attention 层接收 `rope_cos_sin_cache/rope_positions/attn_mask/attn_mask_meta`;新增 `build_varlen_mask_meta_from_lengths/from_ranges` 从 host 侧有效长度直接构建 varlen FA 元数据(避免 GPU `nonzero` 动态 shape 路径),对批内填充部分做 attention mask。性能上新增两个 Triton 融合核(`zimage_native_norm.py`):bf16 舍入逐步对齐原生的 rmsnorm+scale 与 rmsnorm+tanh(gate)+residual,`quant_config is None` 时启用、不满足条件时回退 PyTorch 组合;RoPE cos/sin 与 attn mask/meta 均按键缓存复用。

## 关键改动

```text
python/sglang/jit_kernel/diffusion/triton/zimage_native_norm.py: bf16 原生舍入的 rmsnorm_scale / rmsnorm_tanh_residual 融合 Triton kernel
python/sglang/multimodal_gen/runtime/models/dits/zimage.py: ZImageRMSNorm、批处理 per-sample RoPE、attn mask/meta 构建与缓存
python/sglang/multimodal_gen/runtime/layers/attention/layer.py: build_varlen_mask_meta_from_lengths/from_ranges(host 侧长度构建 varlen 元数据)
python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py: enable_autocast=False、prepare_sigmas 对齐原生
```

```python
class ZImageRMSNorm(nn.Module):
    """RMSNorm that preserves Z-Image's native bf16 behavior.
    Z-Image does not upcast hidden states to fp32 for RMSNorm."""
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (output * self.weight.to(x.dtype)).to(orig_dtype)
```

## 性能声明

1x H100,1024x1024:

- 单张生成(9 steps):loop time 863.27 ms(main)-> 851.57 ms(本分支)
- 批生成 size 5:Batch Wall 4907.5 -> 4781.3 ms,Mean Client 4893.5 -> 4765.0 ms,Denoising 3749.0 -> 3691.5 ms,Avg Step 416.3 -> 409.9 ms

正文总结:单张与批生成均约快 2%。本 PR 的主收益是精度修复(见下),性能为顺带结果。

## 验证与回归风险

精度验证:1x H100,`--batching-mode dynamic --batching-max-size 5 --batching-delay-ms 1000`,seed 42、8 steps、guidance 0.0,5 个 prompt 的生成图像与原生 PyTorch 实现(`Tongyi-MAI/Z-Image`)截图对比。新增单测(素材省略的 7 个文件中)覆盖 RoPE/mask/融合核路径。风险:归一化数值路径改为 bf16 舍入链,与 SGLang 既有共享 fp32 累加 RMSNorm 输出不同(这是与原生对齐的目的);Triton 融合核仅在 bf16、dim<=8192、可平凡展平 stride 时启用,否则回退;attention mask/varlen 元数据缓存按长度与 device 键控,批组成变化即重建。CI:Base 与 Extra 均失败(素材所示)。

## 关联

修复:#28502(动态 batching 下 Z-Image 图像劣化)。同属 multimodal_gen diffusion 模型特定 fast path;其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
