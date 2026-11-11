# PR #49024: [Perf] [AMD] wan2.2: Gate FlyDSL norm and scale-shift fusion on high quality

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yz-wqf |
| 合并 | 2026-11-10 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | amd, lora, run-ci, diffusion, jit-kernel |
| Diff | +263/-2,7 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/49024 |

## 一句话结论

Wan 2.2 的 FlyDSL norm/scale-shift 融合按请求质量门控,high 档 720p×40 步端到端 553.644→538.630 s(1.028×)。

## 动机

Wan 2.2 的 normalization、gated residual 与 scale/shift 链可以复用 ROCm 上已有的 FlyDSL fused kernel,但这些 kernel 会把部分 FP32 操作数与残差降到 BF16,启用必须尊重请求的数值质量契约。本 PR 让标准 Wan 路径仅在 `SGLANG_USE_ROCM_FLYDSL=1` 且 `quality="high"` 同时满足时才启用该优化;默认 `lossless` 请求即便开了环境 flag 也保持 reference 路径。

## 优化点

注册 `WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT` 到既有的请求级 `QualityGatedFusion` 机制:site 的静态 reject 检查 ROCm 平台、`_use_rocm_flydsl`、eps 必须为 1e-6(FlyDSL kernel 固定 epsilon)、hidden size 对齐 `FLYDSL_NORM_MIN_ALIGNED_DIM` 及 FlyDSL 可用性,任一不满足则整组不 mount(all-or-nothing)。denoising 阶段在 batch 边界为 `high` 请求 mount、`exact`/`lossless` unmount。`WanTransformerBlock` 的 norm1 与三个 residual norm 以 `rocm_flydsl_high_only=True` 标记;norm_out 与未经验证的 VSA(video_sparse_attn)块共用,保持默认不启用。`forward_hip` 在 site 已标记但未启用时走 `forward_native`。不引入新 kernel,复用既有 FlyDSL norm 与 scale/shift kernel(含 gated-residual 变体)。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/sites/wan_rocm_flydsl_norm_scale_shift_site.py: 新增 QualityGatedFusion site(静态 reject 条件)
python/sglang/multimodal_gen/runtime/layers/layernorm.py: mark 与 forward_hip 的 quality 门控
python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py: norm1/三个 residual norm/norm_out 标记(VSA 除外)
python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py: ("high", WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT) 登记
python/sglang/multimodal_gen/test/unit/test_wan_rocm_flydsl_dispatch.py: 新增 dispatch 单测
```

```python
if not _use_rocm_flydsl or (
    WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.metadata(self) is not None
    and not WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.is_enabled(self)
):
    return self.forward_native(residual, x, gate, shift, scale)
```

## 性能声明

单卡 MI350X,顺序冷启 server,Wan2.2 T2V A14B Diffusers,1280×720、81 帧、40 步、seed 42、guidance 4.0/3.0,torch.compile 关闭,两侧均 `quality="high"`,仅 FlyDSL flag 不同:客户端端到端 553.644 s→538.630 s,降 15.014 s(2.71%),即 1.028× speedup。历史 kernel-only 同进程探针:7800×5120 0.2950→0.0884 ms(3.34×)、75600×5120 2.6025→0.7262 ms(3.58×)(FlyDSL 返回 BF16,native 返回 FP32)。输出相似度(以 native high 为参考,非 ground truth):Wan 2.2 T2V A14B 1280×720 PSNR 16.7063 dB/SSIM 0.647476;Wan 2.1 T2V 14B 832×480 PSNR 15.3451 dB/SSIM 0.511038。

## 验证与回归风险

运行时插桩验证实际 dispatch:默认 lossless(1 步探针)121 个标记站点全 native;`high`+flag(40 步)242 个站点全 FlyDSL;`high` 无 flag 全 native。新单测覆盖质量切换恢复 native、五种 reject 原因的 all-or-nothing、环境权限在层构建前缓存;LoRA adapter 安全性登记到 `_ADAPTER_SAFE_FUSIONS`("norm, modulation and residual only; no linear")。风险:FlyDSL 输出 BF16,属近似路径,仅限 `high` 档;Wan-Animate-2 的自定义 denoising 绕过 quality mounting,不激活;VSA 路径未验证。

## 关联

复用既有 FlyDSL kernel 与 QualityGatedFusion 机制;同批 #41794 亦走 ROCm FlyDSL 融合(Kimi-K3 KDA decode)。素材未提及本 PR 的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
