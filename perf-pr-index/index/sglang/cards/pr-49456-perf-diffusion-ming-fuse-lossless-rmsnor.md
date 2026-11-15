# PR #49456: [Perf] diffusion/ming: Fuse lossless RMSNorm scale and residual epilogues

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/49456 |

## 一句话结论

Ming 无量化 DiT 的 BF16 scale/residual epilogue 融进现有 ordered RMSNorm kernel,每个 12-step 请求移除 2,304 次 pointwise launch;B300 SM103 上 E2E 配对加速 1.035756x(−3.452%,95% CI [1.034145, 1.037420]),全部输出字节与 RGBA 像素与 native reference 一致。

## 动机

Ming 的 unquantized DiT 已用 ordered RMSNorm,但其后仍单独 launch BF16 scale 与 residual 操作。本 PR 把这些 epilogue 融进现有 ordered kernel,每个 12-step 请求移除 2,304 次 pointwise launch,同时保留原有 intermediate rounding。

## 优化点

- 共享 ordered kernel(`_ordered_norm_kernel`)增加可选 Scale/Residual operands:保留 FP32 statistics、learned-weight 乘法前的 cast、scaling 前的 cast、residual 加法前的 cast;`tanh` 保持 native。
- Ming-specific block hooks:新增 `MingImageTransformerBlock`(override `_norm_scale` / `_norm_tanh_mul_add`),Z-Image 的默认 helpers 保留(`ZImageTransformerBlock` 增加同名 hook 方法,模型经 `block_cls` 类属性选择 block 类)。
- Eligibility:`_ming_norm_scale` 在 CUDA、BF16、width-3840(`x.ndim==3`、`shape[-1]==3840`、`16 <= rows`、`rows < 2^29`、scale shape `(B,1,3840)` 同 dtype/device 且 last-dim contiguous、residual contiguous)时经既有 per-signature bit-exact gate(`_MING_NORM_SCALE_FUSION`)走新 custom op `rmsnorm_cast_before_weight_scale_bitexact`;unsupported layouts、small reductions、quantized blocks(`enable_ming_norm_fusion = quant_config is None`)与 compilation(`torch.compiler.is_compiling()`)保留原表达式;`variance_epsilon` 仅接受 1e-6/1e-5。
- 无新 user flag;cookbook 更新说明无量化 BF16 DiT 自动融合、保留 lossless 结果。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/norm/rmsnorm_cast_before_weight_bitexact.py: _ordered_norm_kernel 增加 Scale/Residual 分支;新 custom op rmsnorm_cast_before_weight_scale_bitexact + can_use 判定
python/sglang/kernels/ops/diffusion/__init__.py: 注册 diffusion.rmsnorm_cast_before_weight_scale(TRITON, CUDA)与别名映射
python/sglang/multimodal_gen/runtime/models/dits/ming_image.py: MingImageTransformerBlock + _MING_NORM_SCALE_FUSION(per-signature gate)
python/sglang/multimodal_gen/runtime/models/dits/zimage.py: block 增加 _norm_scale/_norm_tanh_mul_add hooks;block_cls 类属性
docs/cookbook/diffusion/inclusionAI/Ming-Image.mdx: 融合说明与 B300 验证范围
python/sglang/multimodal_gen/test/unit/test_ming_image.py: epilogue 一致性 + fallback 回归测试
```

```python
result = mul_rn_f32(normalized, weight[None, :])
if Scale is not None:
    scale = tl.load(
        Scale + (rows // SEQ)[:, None] * SCALE_STRIDE + cols[None, :], mask, 0
    ).to(tl.float32)
    result = mul_rn_f32(result.to(tl.bfloat16).to(tl.float32), scale)
    if Residual is not None:
        residual = tl.load(
            Residual + rows[:, None] * DIM + cols[None, :], mask, 0
        ).to(tl.float32)
        result = residual + result.to(tl.bfloat16).to(tl.float32)
tl.store(Out + rows[:, None] * DIM + cols[None, :], result, mask)
```

## 性能声明

- 环境:一台固定 NVIDIA B300 SXM6(SM103,driver 595.91.07,`GPU-97c2fe64-...`);native SGLang backend、BF16 模型执行、FP32 CPU latents;1024×1024、12 steps、CFG1、seed42、batch/concurrency 1;全部组件常驻;Torch 2.14.1+cu130、Triton 3.8.0、Diffusers 0.37.0、Transformers 5.19.0、FlashInfer 0.7.3;计时覆盖 `DiffGenerator.generate` 到 RGBA materialization。三轮 main–PR–PR–main 独立轮次,120 measured / 144 total;配对取每 index arithmetic arm means 的比值再几何平均。
- **最终配对结果:1.035756x、E2E 降低 3.452%**,经验 hierarchical-bootstrap 95% CI [1.034145, 1.037420]x;预注册 A/A 进程内最大 range 1.596%,两个 prompt strata 均超过它;轮次比值 1.037371x / 1.034585x / 1.035313x。
- 分 prompt(short/long 各 30 measured requests per arm):native denoise p50 main→PR 1.2004→1.1486 s(short)/ 1.2504→1.1984 s(long);main E2E p50/p95 1.3523/1.3656 → 1.3042/1.3247 s(short)、1.4072/1.4231 → 1.3588/1.3732 s(long);paired E2E reduction 3.459% / 3.446%;serial 1/mean latency 0.7385→0.7649 / 0.7098→0.7351 req/s;peak reserved 52070 → 52070 MiB。denoise 列用 native 异步 stage timer;同步 GPU interval 保留在单独 instrumented profiles、不替代 benchmark 时序。
- Profile 机制:eager CUDA kernels 14,614 → 12,310;BCG replay trace 15,031 → 12,727;两者各移除 1,536 次 BF16 multiplies 与 768 次 additions(合计 2,304/请求,与动机一致);ordered norm、tanh、attention/GEMM 等其余 kernel counts 不变;non-denoise stages 不变。eager auditing 每请求 768 scale + 768 residual custom-op calls、verification gate 开启。
- Cleanup retest:两轮 prototype–final–final–prototype(80 measured / 96 total)1.000817x(95% CI [0.998465, 1.003343]),0.082% 在噪声内,无 cleanup 回归;final-head 结果独立于 prototype 筛选。
- BCG startup diagnostic(非速度声明):main 初始化 33.300 s / `_capture` 0.470 s / 35 segments / 52838 MiB;PR 33.161 s / 0.465 s / 35 / 52838 MiB。
- vLLM-Omni 对照(frozen native `725c1c9a`,同 checkpoint、同 GPU):native default 的 RGB SSIM 0.7060/0.7192、PSNR 15.60/15.83 dB,未通过预声明质量 gates(SSIM≥0.95、PSNR≥28dB、CLIP≥0.92),既有 SGLang FA 替代亦未过 short-prompt RGB gate——因此不声明任何 quality-qualified Omni speedup。

## 验证与回归风险

- 同一物理 B300 上 48 个 Ming tests 与六个共享 RMSNorm regressions 通过;六个 epilogue 用例覆盖 odd lengths、batch two、strided scale views、changed inputs、对独立 native RMSNorm 表达式的 CUDA graph replay;compile fallback 用 `torch.compile(backend="eager", fullgraph=True)` 实测真实 Dynamo tracing(非 Inductor 性能结果);六文件 pre-commit(Ruff 0.15.1 / isort 7.0.0)与 Mintlify build/link checks 通过。
- Final-head 验证:24 组 cross-arm pair comparisons + 7 次 repeat checks(含 pre-cleanup prototype);每个完整 `[4,1024,1024]` BF16 输出字节与 RGBA 像素全部匹配;144 个 final timing 输出与 96 个 cleanup timing 输出同样匹配 retained native reference;无样本剔除。
- 既有 BCG 兼容性:native warmup 捕获字面 `warmup` prompt;seed 42、seed 43、lossless revisit 各执行 12 次 runner replays;short/long shapes 与 high mode 走预期 eager fallback;无 serving captures;replay 内 GPU trace 显示同样的 2,304 次移除调用;不改 warmup builder、不新增 profitable BCG scenario。
- 范围限制(正文自述):仅一个 B300 上 Design text-to-image 的上述配置;Design-Layer、image editing、multi-GPU、quantized weights、其他架构与 first-use performance 未由该模型级 benchmark 验证;共享 kernel regressions 与 first-sight fallback 保护既有 call sites,但不替代模型级测量。

## 关联

无(正文未列出关联 PR;vLLM-Omni 对照为 frozen commit `725c1c9a`,非 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
