# PR #26045: Apply apply_group_norm_silu to LTX-2 latent upsampler

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @pengdurice |
| 合并 | 2026-06-02 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | pack 未提供 diffstat(diff 完整可见,新增 1 个测试文件、改 3 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/26045 |

## 一句话结论

将 LTX-2 latent upsampler 的 9 处 silu(norm(x)) 接入 fused GroupNorm+SiLU Triton kernel,LTX-2 形状内核提速约 10.7x。

## 动机

LTX-2 两阶段 pipeline 的 spatial `LatentUpsampler` 位于每次生成的关键路径上:Lightricks 先在低分辨率生成 latent,再由该模块 2x 上采样后送 VAE decode。其 `LatentUpsampler` 与 `ResBlock` 多处使用 `GroupNorm(32)+SiLU`,但现有代码是 eager 的 `silu(norm(x))`,绕过了 HunyuanVAE 已经接入的 Triton fused `apply_group_norm_silu` helper。PR 同时呼应 [Roadmap] SGLang-Diffusion 26Q2(#23035)中"kernel fusion instead of compiler-driven fusion"一项。

## 优化点

不写新内核,而是把 `latent_upsampler.py` 中三处符合条件的直接 `silu(norm(x))` 站点接入既有 helper:forward 的 initial_norm/initial_activation(2D 与 3D 路径各 1 处)与每个 ResBlock 的 norm1/activation。默认 `num_blocks_per_stage=4` 时每次 upsampler 调用命中 9 个融合站点(1 initial + 前 4 + 后 4 ResBlock)。ResBlock 内 `silu(norm2(x)+residual)` 因 helper 不支持带 residual 的模式而刻意留在 eager 路径,留待后续 `apply_group_norm_add_silu` kernel(每代还有 8 个站点,可再翻倍覆盖)。性能来源的分解:fusion 贡献恒定约 1.55x(省一次全张量读写),chunked Welford 并行化贡献 4x-9x——PyTorch `native_group_norm` 每 (batch, group) 只启 1 个 block,32 组在 132-SM H100 上仅约 24% SM 占用;生产 `_launch_chunked` 路径把每组拆成 8 个 192 元素 chunk 并行 reduce,post_720p 约 57600 个 program,达到约 2.87 TB/s(86% H100 HBM 峰值)。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/upsampler/latent_upsampler.py: 三处 silu(norm(x)) 改走 apply_group_norm_silu
python/sglang/jit_kernel/benchmark/diffusion/bench_group_norm_silu.py: 新增 3 个 LTX-2 upsampler 形状 case(post_720p 约 940 MB,需 H100/H200 级显存)
python/sglang/multimodal_gen/test/unit/test_latent_upsampler_group_norm_silu.py: 新增 19 个单元测试
```

```python
# Fused GroupNorm + SiLU on the first norm of the block. The second
# norm is followed by `silu(norm + residual)`, which the
# current `apply_group_norm_silu` helper does not cover -- left on
# the eager path until a `group_norm_add_silu` helper exists.
x = apply_group_norm_silu(x, self.norm1, self.activation)
```

## 性能声明

测试条件:H100 SXM5(p5.48xlarge),`bench_group_norm_silu.py`,warmup 25 / rep 100 / 3 rounds,num_groups=32。

| case | dtype | shape | native (µs) | fused (µs) | speedup |
|---|---|---|---:|---:|---:|
| ltx2_upsampler_small | bf16 | 1x512x8x45x80 | 452.00 | 70.05 | 6.45x |
| ltx2_upsampler_pre_720p | bf16 | 1x512x16x90x160 | 3468.27 | 261.87 | 13.24x |
| ltx2_upsampler_post_720p | bf16 | 1x512x16x180x320 | 13710.72 | 987.52 | 13.88x |

fp16 结果相近(6.52x/13.33x/13.93x);LTX-2 upsampler 形状几何平均内核提速约 10.7x。E2E 估算:720p clip 每次 `LatentUpsampler.forward` 9 个站点共省约 67 ms(bf16),占 upsampler 自身 runtime 预算的 20-45%;PR 说明相对整条多秒级生成的 E2E 增益较小。

## 验证与回归风险

19 个单元测试:CPU fp32 零容忍 parity(atol=0, rtol=0,与 eager 参考逐位一致)+ 融合站点计数断言(每 ResBlock 恰 1 次调用、upsampler 共 1+2*num_blocks_per_stage 次);CUDA bf16/fp16 生产路径 parity(bf16 atol=7e-2/rtol=2e-2,fp16 e2e 放宽到 atol=2e-2/rtol=1e-1 以吸收下游 8+ 层 conv 的漂移),并用 patch 断言实际走到 `triton_group_norm_silu` 而非静默回退 eager。Triton kernel 本身已有 `jit_kernel/tests/diffusion/test_group_norm_silu.py` 覆盖。回归面:LTX-2 全部生成路径(upsampler 在关键路径),输出数值有 kernel 级舍入差异;bench 的 post_720p case 被标注 disabled="standalone benchmark",不进 CI 内存预算。

## 关联

前序:同一 helper 在 HunyuanVAE 的接入(pack 正文提及);#23035 为 [Roadmap] SGLang-Diffusion 26 Q2。后续方向:PR 预告 `apply_group_norm_add_silu` kernel 以覆盖 ResBlock 第二处 norm 站点。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
