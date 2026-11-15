# PR #49391: [Perf] [NVIDIA] diffusion/joy: Reuse FP32 residual gate for large image updates on SM103

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-11-12 |
| 分类 | Other(tier: major) |
| Label | documentation, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/49391 |

## 一句话结论

Joy Image Edit 的大 image residual 更新(每请求 4,800 次 `torch.addcmul`)改用既有 `residual_gate_fp32` primitive、把 FP32 乘积保留到最终 BF16 cast;B300 SM103 上 E2E 配对加速 1.016774x(延迟 −1.6497%,bootstrap 95% CI [1.016186x, 1.017824x]),全部输出与 main 逐字节/逐像素一致。

## 动机

Joy Image Edit 用 `torch.addcmul` 做 image 与 text residual 更新,separate model profile 显示每请求 9,600 次调用。本改动把其中 4,800 次 large image updates 换用现有 `residual_gate_fp32` primitive,FP32 product 保留到最终 BF16 cast;per-signature 的 `BitExactFusionGate` 先对实际 native reference 校验,通过才保留 fast path。

## 优化点

- `fused_add_gate` 重写:eligible 时走 `diffusion_kernels.residual_gate_fp32` + per-signature gate `_JOY_RESIDUAL_GATE`。
- Eligibility(全部满足才走 fast path):contiguous BF16 `[B,S,D]` 输入、至少 16,777,216 elements、SM103;不在 compiling、非 grad-enabled、CUDA、`residual.shape == x.shape`、三张量同 dtype/device、contiguous、gate shape `(B, 1, D)`。小 text updates、不支持的 shapes/layouts/devices、autograd 与 compilation 保留 native `torch.addcmul` 路径。
- 三文件改动,无新 kernel、无新 flag、无 model-level BCG 支持;fused-kernel 文档与 focused regression tests 同步更新。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/joy_image.py: fused_add_gate 增加 SM103 大 tensor fast path + _JOY_RESIDUAL_GATE(per-signature bit-exact gate)
docs/docs/sglang-diffusion/fused_kernels.mdx: residual_gate_fp32 条目与 Joy 用法说明
python/sglang/multimodal_gen/test/unit/test_joy_image_residual.py: 新测试文件(大 shape + CUDA graph replay + 梯度 fallback)
```

```python
def fused_add_gate(residual, x, gate):
    """Preserve addcmul's FP32 product before the final cast."""
    gate = gate.unsqueeze(1)
    def reference():
        return torch.addcmul(residual, x, gate)
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or not residual.is_cuda
        or not current_arch().is_sm(10, 3)
        or residual.numel() < 16 * 1024 * 1024
        or residual.ndim != 3
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or gate.shape != (residual.shape[0], 1, residual.shape[2])
    ):
        return reference()
    return _JOY_RESIDUAL_GATE.run(
        lambda: diffusion_kernels.residual_gate_fp32(residual, x, gate),
        reference,
        sig=(residual.device, residual.shape, gate.stride()),
        logger=logger,
    )
```

(摘录省略了 guard 中其余 shape/dtype/device 等价条件,完整条件见 pack diff。)

## 性能声明

- 环境:一台物理 NVIDIA B300 SXM6 AC(UUID `GPU-97c2fe64-...`);native SGLang backend;BF16;1024×1024、30 steps、CFG4、seed42、CPU RNG、max sequence length 4096、batch/concurrency 1;两计时臂均为 eager(无 BCG、无 offload)。测量前对 main / pre-cleanup / cleaned source 各做 A/A:每进程 2 warmups + 6 平衡测量,单 prompt 全 range 最大 0.347698%,全部通过 0.5% gate;48 个 A/A native 输出与像素精确一致。
- 表(Main `5325cfabd7` vs PR `0b6e700701`,各 36 measured requests):E2E median/p95 9.8691/9.9146 s → 9.7072/9.7401 s;Denoise median(异步诊断)9513.18 → 9347.19 ms;Peak reserved 均为 54184 MiB。三轮新 ABBA(12 fresh processes、72 measured requests)给出 **1.016774x paired speedup / 1.6497% lower E2E**,round-cluster bootstrap 95% 区间 [1.016186x, 1.017824x];每轮/每 prompt 均改善,无样本剔除;统计基于每 process/prompt median 的几何平均配对比值。
- 全部 96 个 final timing 输出(含 24 warmups)native bytes 与 RGB pixels 完全一致。E2E 含 native generation 与 pixel materialization,不含 hashing/saving/PNG 编码;不做 HTTP serving 吞吐或并发声明。
- Profile 机制(matched synchronized profiles;GPU denoising stage 在两条 trace 中跨 9704.50 / 9507.00 ms,trace 用于机制说明、非 latency 样本):4,800 次 large-image native addcmul 一对一替换为 `_residual_gate_fp32_kernel`;4,800 次 text addcmul 保留。Native addcmul(image+text)9600 calls / 576.38 summed GPU ms → 4800 / 77.53;large-image FP32 residual primitive 0/0 → 4800 / 137.06;其余 per-kernel counts 不变。
- Cleanup 中立性:两轮 prototype–final ABBA(80 measured / 96 total)0.999541x(+0.0459%),落在预定 [0.995, 1.005] 中性区间内;更早一次 40-measurement cleanup 尝试未过 gate(round0 long-prompt ratio 1.007272953),已归档、不与新系列合并。
- Native vLLM-Omni 对照(Omni 自有 Joy pipeline,同 GPU/checkpoint/输入;eager 与 40-block regional compile 两种)均未通过预注册的跨框架质量 gates(SSIM≥0.95、PSNR≥28dB、CLIP cosine≥0.92),故对 Omni 不声明任何延迟比或聚合收益;本 PR 的性能声明只针对 frozen SGLang main。

## 验证与回归风险

- 精确 committed head 通过 18 组 main/PR 请求对:eager lossless(seed 42/123/2026)、eager high(seed 42)、native BCG lossless/high(seed 42),各带 short/long/short 请求复用;每个完整 FP32 native `[3,1,1024,1024]` 张量(3,145,728 elements)与 RGB 像素与对应 main run 完全一致(max error/RMSE 0、SSIM 1、PSNR 无穷)。12 个 eager quality requests 各验证 4,800 次实际 fusion 调用、签名核实、无 disabled gate。
- 三个 focused tests:实际 large image shapes(含 batch2)、CUDA Graph replay 下的 changed inputs、CPU/CUDA 梯度 fallback;另一测试以 `fullgraph=True` 编译最终 wrapper、用 failure sentinel 替换 eager fusion primitive、对照 compiled native `addcmul` 验证两轮 changed-input 迭代(wrapper fallback 兼容性,非全模型 compile 性能)。
- BCG checks 保持对应 native-main 输出与 lifecycle counters(lossless/high 的 short/long/short counters 与 main 一致);native baseline serving misses 与 BCG-versus-eager 输出差异仍是局限;本 PR 不新增 BCG 覆盖、不声明 BCG 加速。High mode 在此 SM103 workload 上不激活 SM120-only GELU fusion 或 SM90-only QK fast path。历史部分计时(partial formal-v1 等)不作为 final-head 结果。

## 关联

#40265、#48366、#49189(正文引用的 source/review examples);对照用 frozen vLLM-Omni commit `725c1c9a`(非 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
