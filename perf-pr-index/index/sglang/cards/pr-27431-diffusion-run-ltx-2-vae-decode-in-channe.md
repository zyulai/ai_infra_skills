# PR #27431: [diffusion] Run LTX-2 VAE decode in channels_last_3d (faster decode, lower peak memory)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-09 |
| 分类 | Model-specific Fast Path | tier: major |
| Label | run-ci, diffusion, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 4 个文件:`vae_loader.py`、`ltx_2_vae.py`、新增 `test_ltx2_vae_channels_last.py`、`test_vae_loader.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/27431 |

## 一句话结论

LTX-2 VAE decode 改跑 channels_last_3d 内存格式,decode 阶段 1.41x、conv 微基准最高 4.58x,峰值显存降 9.7 GiB。

## 动机

LTX-2 video VAE 的 decode 阶段是 conv3d-bound。Hopper 上 `Conv3d` 在 `channels_last_3d`(NDHWC)内存格式下快约 2.7-3.7x,且 VAE loader 已对 Wan/QwenImage 通过 `_convert_conv3d_weights_to_channels_last_3d` 启用该加速,但 LTX-2 一直没享受到,原因有二:一是 `LTX2VideoCausalConv3d` 对 layout 不友好——temporal causal padding 用 `repeat()` + `concatenate()` 实现,产出的 tensor 是 contiguous (NCDHW),直接破坏 `channels_last_3d`,迫使每次 conv 前做昂贵的 layout 重转换,把收益抵消;二是 loader 的 `auto` 策略只对 QwenImage 与单 GPU Wan 启用 `channels_last_3d`,把 LTX-2 排除在外。

## 优化点

两处改动。其一,layout-preserving causal pad:当 conv weight 已是 `channels_last_3d` 时,padded tensor 直接按 `channels_last_3d` 分配并用 `copy_` 填充,一次 allocate-and-copy 同时完成 replication pad 与 layout 修正,避免 `repeat`+`concat` 及 NCDHW→NDHWC transpose 中间体——这也是峰值显存下降 9.7 GiB 的来源。该路径按 weight layout 自门控:`channels_last_3d` 关闭时,原 `repeat()`+`concatenate()` 路径逐字节不变,默认路径零风险。`copy_` pad 本身是 bandwidth-bound(最大 decode shape 约 0.5 ms,接近硬件极限),作者原型化过自定义 Triton pad kernel 并实测未能更快,故不引入。其二,loader `auto` 策略对单 GPU LTX-2 启用 `channels_last_3d`(与 Wan 同约定;multi-GPU 仍需显式 `SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=1` 覆盖)。本 PR 与 #26878 互补:后者优化 decode 阶段编排(untiled-decode policy + GPU-side postprocess)但不改 conv 内存格式;两者可叠加,`channels_last_3d` 同时加速 tiled 与 untiled 路径中的每个 conv。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/vaes/ltx_2_vae.py: 新增 _causal_temporal_pad_channels_last() 与 weight layout 检测,pad 直接在 NDHWC 完成
python/sglang/multimodal_gen/runtime/loader/component_loaders/vae_loader.py: _should_use_channels_last_3d 把单 GPU LTX2PipelineConfig 纳入 auto 策略
python/sglang/multimodal_gen/test/unit/test_ltx2_vae_channels_last.py: 新增;pad/conv 与原路径等价性测试
python/sglang/multimodal_gen/test/unit/test_vae_loader.py: LTX-2 单/多 GPU gate 用例
```

```python
out = torch.empty(
    (b, c, t + left + right, h, w),
    dtype=x.dtype, device=x.device,
    memory_format=torch.channels_last_3d,
)
out[:, :, left : left + t].copy_(x)
if left:
    out[:, :, :left].copy_(x[:, :, :1] if left_pad is None else left_pad)
if right:
    out[:, :, left + t :].copy_(x[:, :, -1:] if right_pad is None else right_pad)
```

## 性能声明

- 微基准(H100,bf16,isolated):`conv3d` alone(C256,64x128x192)31.3 ms→8.5 ms(**3.67x**);完整 `LTX23VideoDecoder3d.forward` 24.8 ms→5.4 ms(**4.58x**)。
- 端到端(H100 单卡,native SGLang diffusion,`LTX2TwoStagePipeline`,LTX-2 snapshot 47da56e2,1536x1024、121 帧、fps 24、`--enable-torch-compile`,与 #26878 同命令同方法,两侧均 warm 测量):Decode 5.41 s→3.84 s(**1.41x,-29%,-1.57 s**);Denoise 36.89→36.63 s、Refine 7.33→7.24 s 均不变(DiT 无 Conv3d);E2E(warmup 除外)52.55→51.77 s;**Peak reserved 71.81 GiB→62.12 GiB(-9.7 GiB,-13.5%)**。
- 作者解释:E2E 1.41x 低于 isolated 3.67x,因为默认 decode 是 tiled 路径,且该阶段含非 conv 工作(norms、upsample reshapes、audio VAE、postprocess、CPU-offload 传输);配合 #26878 的 untiled decode,conv 占比与收益会增大。
- 输出视频与 baseline 视觉等价(同 seed 同 denoise)。

## 验证与回归风险

测试:`test_ltx2_vae_channels_last.py` 断言新 pad 与原 `repeat`+`concat` 等价(causal / non-causal / temporal-only kernel,edge-frame replication 精确,`channels_last_3d` 在 CUDA 上经 conv 保持);`test_vae_loader.py` 覆盖单 GPU LTX-2 启用、多 GPU 关闭的 gate。数值正确性:pad 与原路径数值相同(replicate-edge == repeat-edge frame,测试断言 bit-exact);唯一差异是 cuDNN NDHWC conv kernel 的浮点累加顺序——与 Wan/QwenImage 已上线的权衡相同,module 级 fp32 decode diff 约 1e-4。风险:bf16 输出存在累加顺序差异(视觉等价);默认(非 channels_last)路径逐字节不变;多 GPU LTX-2 默认仍关闭;流式 conv_cache 路径(padding 帧来自上一 chunk)同样走 layout-preserving 分支,有专门等价测试。

## 关联

#26878(同区域前序:decode 阶段编排优化,untiled-decode policy + GPU-side postprocess,与本 PR 可叠加)。无正文列出的其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
