# PR #46722: [Perf] [AMD] diffusion: Optimize the Wan VAE decode for gfx1250

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-10-29 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | amd, run-ci, diffusion |
| Diff | pack diffstat 缺失;diff 覆盖 rocm.py、wanvae.py、wan_vae_cuda_opt.py、configs wanvae.py 及两个测试文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/46722 |

## 一句话结论

gfx1250(MI450)上 Wan VAE decode 从 22.5 s 降到 8.5 s(NHWC conv + fused norm fast path + 去 math SDPA 的 where),TP2 decode 28.5 → 9.0 s。

## 动机

torch profile 显示 Wan VAE decode 约 20 s GPU 时间中只有 ~9 s 是卷积,其余是未融合的 fp32 elementwise 与一处 attention fixup:`RocmPlatform.optimize_vae` 从未安装 `wan_vae_cuda_opt`,每个 `WanRMS_norm → SiLU` 跑成 `linalg_vector_norm + div + mul + silu`(每次 decode div 4.8 s、mul 3.0 s、silu 2.7 s)。fused norm 需要 channels_last_3d 输入,而 ROCm batched-Conv2D replacement 产出 `[N, T, C, H, W]` 内存;PyTorch 仅在 `PYTORCH_MIOPEN_SUGGEST_NHWC=1` 时以 NHWC 运行 MIOpen convs,否则 channels_last conv2d 会被转置到 NCHW 再转回。gfx1250 将 VAE attention 钉在 math SDPA backend(AOTriton 在 head_dim>256 失败),每 call 131 ms 中 121 ms 是 `_safe_softmax` 里清零全掩码行的 `where`——此处无 mask。多 GPU 下 `auto` 模式在 gfx1250 选择 spatial-shard decode 并跳过 fast path,2 ranks 上 28.5 s 反而比 1 rank fast path 慢。

## 优化点

四项机制改动,全部以 `is_gfx1250_supported()` 门控,NVIDIA/MI300/MI355 路径不变:(1) gfx1250 上 `optimize_vae` 安装 `maybe_optimize_wan_vae` / `maybe_optimize_qwen_image_vae` fast path(fused RMSNorm+SiLU、DupUp3D+add);(2) conv replacement 保留 channels_last BF16 2-D 权重并以 `setdefault` 设置 `PYTORCH_MIOPEN_SUGGEST_NHWC=1`,`_conv3d_as_batched_conv2d` 增加 NHWC 分支:一次 copy 把 Kt 帧 gather/cast 进 `[N*T_out, H, W, Kt*C_in]`,输出免拷贝折叠回 channels_last_3d;(3) `_sgl_math_sdpa` 且 fp32 时用显式 math attention(`q*s @ (k*s)^T` → softmax → `@ v`),与 math backend bit-identical 并跳过 `_safe_softmax` 的 `where`;(4) 多 GPU:`auto_parallel_decode_prefers_spatial_shard()` 在 gfx1250 返回 False,`use_parallel_decode` 仅在选中 spatial-shard 时为 True,cfg-parallel 下只有主 rank decode。`wan_vae_cuda_opt.py` 放开 HIP 上的 exact-tier post-op fusion,保留首次使用 `torch.equal` 校验、不匹配回退 aten。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/platforms/rocm.py: gfx1250 安装 VAE fast path;NHWC conv2d 分支与 MIOPEN NHWC flag
python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py: gfx1250 引入 Triton VAE helpers;显式 math attention;auto 模式 use_parallel_decode 门控
python/sglang/multimodal_gen/configs/models/vaes/wanvae.py: auto_parallel_decode_prefers_spatial_shard() 返回 not is_gfx1250_supported()
python/sglang/multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py: 放开 HIP exact-tier post-op fusion
```

```python
if getattr(self, "_sgl_math_sdpa", False) and q.dtype == torch.float32:
    s = (q.size(-1) ** -0.5) ** 0.5
    attn = torch.matmul(q * s, k.transpose(-2, -1) * s)
    x = torch.matmul(torch.softmax(attn, dim=-1), v)
else:
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

## 性能声明

Wan2.2-T2V-A14B,720p / 193 frames,fp32 VAE decode,单 gfx1250 partition(MI450),cookbook flags(`SGLANG_USE_ROCM_VAE_CONV2D_BF16=1` 等),单请求:main `3dce594de3` denoise 199.3 s / VAE decode 22.5 s / e2e 223.3 s → 本 PR 198.9 / 8.5 / 208.9 s。kernel 层面(torch profiler):div/mul/SiLU 全部 fused,math SDPA 3.28 s → ~0.2 s,conv NHWC 每 conv 快 1.4–2.9x(34.97 → 11.89 ms @ 96 ch, 720x1280)。TP2(与 #46723 同分支、dashboard shape 90000 DiT tokens,绝对时间与 TP1 表不可比):VAE decode 28.5 → 9.0 s,e2e 124.7 → 105.0 s,TP2 e2e 1.19x faster,TP2 vs TP1 从 1.37x 到 1.63x;PR 注明剩余 denoise gap 是 package power limit(1600 W socket cap),非本代码路径。

## 验证与回归风险

测试:`test_rocm_conv3d_patch.py`(NHWC conv 匹配原 `WanCausalConv3d` 含 residual/skip_bias、返回 channels_last_3d;显式 math attention 与 math SDPA backend `torch.equal`);`test_vae_spatial_parallel_decode.py`(gfx1250 auto 不 shard、显式 spatial_shard 仍 shard;`use_parallel_decode` 仅 gfx1250 auto 为 False)。精度:全 8-step 视频 vs main PSNR mean 38.45 dB / min 35.64 dB(差异来自 NHWC conv 算法与 fused norm 舍入顺序);attention 改动 bit-identical;TP2 main-rank-only decode 与 both-ranks decode 逐帧 identical。风险:MIOPEN env flag 为进程级 `setdefault`;fused cat+pad 需避开被 Conv2D replacement patch 的 convs(`_weight_2d` hasattr 检查);spatial-shard 与单 rank decode 从不 bit-identical(PSNR 38.7 dB)。

## 关联

#46723(TP2 性能测量所在分支同时包含该 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
