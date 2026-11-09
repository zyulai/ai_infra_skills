# PR #46149: [Perf] diffusion: Speed up Wan's spatial-parallel VAE decode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-24 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 6 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/46149 |

## 一句话结论

多 GPU Wan VAE 解码接入 CUDA fast path 与 channels_last_3d,并做行分片 attention,decode 1.284→0.972 s。

## 动机

多 GPU Wan decode 按帧高分片(`WanDistResidualBlock`、带 halo exchange 的 `SpatialParallelCausalConv3d`),该路径从未接入 CUDA fast path("skipping fast path"),且 Conv3d 权重保持 NCDHW(channels_last_3d 策略只覆盖单 GPU)。Wan2.2 T2V A14B 720p 4xB300 fp32 decode(共 1.32 s)的 profile:每个 conv 前后 cuDNN 自身的 `nchwToNhwc`/`nhwcToNchw` 转换 111 ms;未融合的 RMSNorm+SiLU 链(`BinaryFunctor`、`silu`、`norm`、adds)约 250 ms;4114 个 `direct_copy` kernel 149 ms。mid-block attention 每 rank 聚合整个帧高、对所有 query 行做注意力(21 次 fp32 `fmha_cutlassF`,每次 decode 143 ms),然后只保留本 rank 的行。

## 优化点

三处:(1) `wan_vae_cuda_opt.py` 的 fast path 安装器接受 spatial-parallel 的 residual/resample/attention block 类(`residual_block_cls` 变为类元组),删除"parallel decode 跳过 fast path"的逻辑——spatial-parallel block 只是换入 halo-exchange conv,halo 切分与拼接沿高度进行、与内存布局无关;(2) `vae_loader.py` 让 Wan 配置在任意 GPU 数下把 Conv3d 权重转 channels_last_3d,且多 GPU 时默认只转 decoder(cuDNN TF32 conv 在 channels_last 下舍入不同,I2V 条件 latent 差 ~3e-3 会把 40 步 14B 轨迹带偏;`SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=1` 显式转换整个 VAE);(3) `wanvae.py` 的 attention block 支持 `rows=(first, count)` 参数:spatial-parallel 下每个 rank 只对本 rank 高度分片的 query 行做注意力,k/v 仍是全帧。fast path 属既有 "lossless" 质量档(fp32 运算重结合、不降精度),`quality="exact"` 仍走参考实现。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py: 安装器接受 WanDist* 类,不再跳过并行 decode
python/sglang/multimodal_gen/runtime/loader/component_loaders/vae_loader.py: Wan 任意 GPU 数转 channels_last_3d(多 GPU 仅 decoder)
python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py: attention 支持 rows 行分片(本 rank query 行 × 全帧 k/v)
test/.../test_wan_vae_attention_rows.py: 行范围输出 == 全帧输出对应行
```

```python
if rows is not None:
    first, height = rows
    q = q[:, :, first * width : (first + height) * width]
    identity = identity[..., first : first + height, :]
```

## 性能声明

Decode stage 同步计时,Wan2.2 T2V A14B 720p,4xB300,fp32 decode(各 3 次请求):

| 阶段 | 耗时 (s) |
| --- | --- |
| 基线 | 1.284 / 1.266 / 1.269 |
| + fast path(channels_last + 融合 norm/silu) | 1.042 / 1.000 / 1.000 |
| + 行分片 attention | 0.972 / 0.942 / 0.938 |

同 build 无行分片 attention 对照:1.038 / 1.004 / 1.005 s。另:2 GPU decode main 372 ms → decoder-only 277-283 ms;代价是 encoder 不再获得 channels_last 的约 8% encode 加速。

## 验证与回归风险

Wan2.2 T2V A14B 720p 4xB300:x264 crf 0 存视频,对未改动 decode PSNR 63.2 dB、SSIM 0.99999、max|diff| 4/255(3 请求),分片边界(行 180/360/540)无接缝;行分片 attention 输出 mp4 md5 与全帧 attention byte-identical。encoder bit-identical、`quality="exact"` 输出 md5 等于 main;2-GPU Wan I2V 一致性用例通过(34.6-38.4 dB 各组);Animate cfg-parallel 与 GT bitwise。新增 `test_wan_vae_attention_rows.py` 与 vae_loader 的 channels_last 策略单测;`test_model_fast_paths.py` 适配元组形式(33 个 Wan/Qwen 用例)。回归风险:多 GPU Wan 默认只转 decoder 的策略与 I2V 数值轨迹强耦合,显式 env 覆盖会转整个 VAE 并改变输出;LTX-2/Cosmos3 保持单 GPU 规则不变。

## 关联

同区域:#41309(同一作者,扩散 attention 路径优化)。fast path 安装器机制为既有代码,前序 PR 编号正文未提及。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
