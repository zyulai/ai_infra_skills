# PR #34617: [Diffusion][HunyuanVideo] Fuse eager QKV packing and high-quality QKNorm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-13 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci, diffusion, jit-kernel, run-ci-extra, mergeable |
| Diff | pack diffstat 缺失;diff 含两个新 kernel 文件、hunyuanvideo.py 模型改造、denoising.py 挂载点与一个新测试文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/34617 |

## 一句话结论

HunyuanVideo eager 路径:单个 Triton kernel 完成 image RoPE + image/text QKV 打包,QKNorm 免 contiguous 拷贝,B300 e2e +5.30%(quality=high)。

## 动机

HunyuanVideo 的 eager 前向在 attention 前做一串可融合的小工作:每个 block 对 img/text Q、K 分别执行 `.contiguous()` 拷贝后 RMSNorm、对 image 部分单独做 RoPE、再用三次 `torch.cat` 拼接 image/text 的 Q/K/V。这些操作都是显存带宽型开销。PR 用三条 scoped 路径减少 eager 开销:(1) 一个 lossless Triton kernel 在把 image/text Q、K、V 打包成连续 attention 输入的同时应用 image RoPE;(2) `quality=high` 的 strided QK RMSNorm 路径,不再物化输入拷贝;(3) 复用已有的 quality-gated fused linear+GELU epilogue 处理 Hunyuan MLP。

## 优化点

QKV/RoPE pack 是单 Triton kernel(`_hunyuan_qkv_rope_pack_kernel`):每个 program 处理一个 token × 一组 head,对 image token 按 even/odd 通道应用 RoPE(`fma(-q1, sin, q0*cos)`),text token 只搬运;输出写入单次分配的 `(3, B, S_total, H, D)` storage 的三个 plane,Q/K/V 一次成型。安全性:per-signature 首次使用 bit-exact 验证(`BitExactFusionGate`,不匹配即回退 reference;CUDA graph capture 期间未验证的签名不尝试);torch.compile 下走 reference。QKNorm 路径直接对 strided view 调 FlashInfer 的 `qk_rmsnorm_cute`(绕过公共 wrapper 的 custom-op dispatch),numerically relaxed 的 RMSNorm 与 GELU 仅在 `quality=high` 挂载;FlashInfer CuTe 入口不可用时 all-or-nothing 挂载,不会泄漏部分质量配置。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/triton/hunyuan_qkv_pack.py: 新增,融合 RoPE + QKV 打包的 Triton kernel
python/sglang/kernels/ops/diffusion/hunyuan_qknorm.py: 新增,strided QK RMSNorm(mount/unmount、quality gate)
python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py: forward 改用 _hunyuan_qknorm/_hunyuan_pack_qkv;HunyuanMLP 挂 GELU epilogue
python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py: 注册 QKNorm fusion 的 mount/unmount
```

## 性能声明

B300,HunyuanVideo,848x480x65 frames,seed 42,30 steps,native backend,对照 main `dbebc1deb42b`(B300 验证树与本 PR 最终 Git 树一致):

| configuration | denoise (s) | e2e (s) | vs main eager |
|---|---:|---:|---:|
| latest main eager | 23.7185 | 26.7040 | baseline |
| PR lossless eager | 22.8496 | 25.8321 | +3.66% / +3.26% |
| PR `quality=high` eager | 22.3054 | 25.2885 | +5.96% / +5.30% |
| PR `torch.compile` | 21.6219 | 23.7775 | +8.84% / +10.96% |

`quality=high` 的 denoise 比 PR lossless 路径快 2.38%;PR 注明它仍落后 compile(denoise −3.16%、e2e −6.35%)。compile 首次撞 benchmark driver 的 180 s warmup 上限,以同缓存、900 s 上限重试成功(编译约 11 分钟,测量不含 warmup)。

## 验证与回归风险

输出比较:lossless 路径 main 与 PR 的 MP4 byte-identical(sha256 一致);`quality=high` 对 main eager 全 65 帧 PSNR 43.35 dB、SSIM 0.9851。测试:新 `test_hunyuanvideo_eager_fusions.py`——两个 production QKV shapes(257/31 与 4096/256 tokens)的 QKV/RoPE pack `torch.equal`;quality QKNorm 挂载后 `atol=2e-2, rtol=2e-2`、卸载后 bit-exact;无 CuTe kernel 时保持 unmounted;B300 focused 测试 4 passed;changed-file pre-commit 通过。风险:`quality=high` 是放宽精度路径(默认仍 lossless);bit-exact gate 依赖首用验证与签名匹配;FlashInfer 缺失时静默回退原 contiguous 路径。

## 关联

无(pack 未给出其他 PR 编号;quality gate 与 fused linear+GELU 为 PR 前已存在的机制)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
