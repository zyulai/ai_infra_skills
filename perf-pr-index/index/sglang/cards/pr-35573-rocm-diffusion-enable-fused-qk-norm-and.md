# PR #35573: [ROCm][diffusion] Enable fused qk norm and rope on ROCm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @gurpreet-dhami |
| 合并 | 2026-09-18 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/35573 |

## 一句话结论

把 FLUX.2 的 fused QK-norm+RoPE 与 fused LayerNorm+modulate 路径从 CUDA 放开到 ROCm,并支持 strided QKV 视图免 `.contiguous()` 拷贝。

## 动机

fused QK-norm + RoPE 与 fused LayerNorm + modulate 内核此前仅在 CUDA 上启用,AMD GPU 上的 FLUX.2 无法享受 NVIDIA 侧已有的 eager-mode 优化。另外,单 transformer block 的 packed-QKV unflatten 产生 strided view,in-place fused kernel 无法直接消费,只能靠隐式拷贝兜底。

## 优化点

三条改动:把 `apply_qk_norm` 与 `apply_qk_norm_rope` 的 fused in-place 路径开关从 `_is_cuda` 放宽为 `_is_cuda or _is_rocm`(仍要求 q_eps == k_eps、fp16/bf16、无隐式拷贝等既有条件);`_flux2_norm_modulate` helper 本已通过 `.is_cuda` 张量检查做到平台无关,本 PR 让其在 double 与 single transformer block 上于 ROCm 生效;给 `apply_qk_norm_with_optional_rope` 新增 `allow_strided_qk` 参数,`Flux2ParallelSelfAttention` 与 `Flux2SingleTransformerBlock` 在 ROCm 上传 `allow_strided_qk=True`,让 kernel 原生处理 strided QKV 视图,避免不必要的 `.contiguous()` 拷贝。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/layernorm.py: apply_qk_norm/apply_qk_norm_rope 门控放宽到 ROCm;新增 allow_strided_qk 透传
python/sglang/multimodal_gen/runtime/models/dits/flux_2.py: ROCm 上传 allow_strided_qk=True,启用 fused LayerNorm+modulate
```

## 性能声明

AMD MI350(ROCm),FLUX.2,TP=1、ULLYSSES=8,per-step per-layer kernel 时间(µs):1024×1024 下 Double Stream Text 101.7→8.4(12.11x)、Double Stream Image 131.6→14.7(8.95x)、Single Stream Joint 153.5→15.8(9.72x),估算 E2E uplift ~10%;2048×2048 下 104.9→8.5(12.34x)、257.1→40.0(6.43x)、339.1→44.4(7.64x),估算 uplift ~6%。端到端(FLUX.2-dev,1 GPU,1024×1024,50 步,seed=42):平均 denoise 步时延 283.85→248.37 ms(约快 12.5%),E2E 16331.11→15179.06 ms(约快 6%)。

## 验证与回归风险

用 SGLang 内置 diffusion 一致性测试 `test_server_1_gpu.py -k flux_2_image_t2i` 验证:CLIP score 0.9843 vs 0.9837(持平或略好),SSIM/PSNR/mean abs diff 跨 build 在噪声范围内(SSIM/PSNR 数值为 AMD-vs-H100 ground truth 对比,相关对比是同平台跨 build)。风险面:仅改 multimodal_gen diffusion 路径的门控与参数透传;strided QKV 路径只在 ROCm 上默认打开,CUDA 路径行为不变;fused 路径仍受 dtype/eps/contiguity 等既有守卫约束。

## 关联

无(素材未列出具体关联 PR 编号;CUDA 侧已存在同型 fused 路径,本 PR 为平台对齐)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
