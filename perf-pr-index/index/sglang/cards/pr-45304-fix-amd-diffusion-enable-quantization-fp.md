# PR #45304: [Fix] [AMD] diffusion: Enable `--quantization fp8` on gfx1250 (torch.compile lowering + per-channel weight scales)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-10-22 |
| 分类 | Quantization (tier: major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,涉及 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/45304 |

## 一句话结论

gfx1250 上启用 diffusion FP8:修复 Inductor 对 rank-0 scale 的 lowering 崩溃,并以 A8W8 Triton tile 把 FLUX.2-dev 720p 端到端加速 1.33x。

## 动机

gfx1250 上 `--quantization fp8` 不可用:开 `--enable-torch-compile` 后首个 warmup 请求即在 Inductor lowering 崩溃(`aten._scaled_mm.default` 对 `args[3]`(weight scale)的 `len(index)==len(stride)` 断言失败)。根因是 ROCm 上 `cutlass_fp8_supported()` 恒为 False,`process_weights_after_loading` 落到 `input_to_float8`,产出 0 维 per-tensor scale——**崩溃与精度损失是同一个 bug**:per-tensor scale 既无法被 Inductor lower,又丢掉了 BF16 checkpoint 在 FP8 下存活所需的 per-channel range。且 hipBLASLt 无 tuned gfx1250 内核,FLUX.2-dev 720p shapes 下 `torch._scaled_mm` FP8 仅 471–506 TFLOPS,低于它要替换的 BF16 torch.compile templates(645–710)。

## 优化点

两处修改、两个目的,任一单独都能止崩溃:

1. **精度修复**:`process_weights_after_loading` 对非 FP8 序列化 checkpoint 走 per-channel 分支(`per_token_group_quant_fp8`),得到真实 `[N]` scale。
2. **性能修复(A8W8)**:新 Triton tile——两个操作数均为 e4m3,激活按 per-token 动态 scale、权重按 per-channel scale、FP32 accumulate,完全绕开 `aten._scaled_mm`;实测 1150–1255 TFLOPS。tile 对 load 按 M/N 取模环绕而非 mask(masked load 在该目标上损失约 10%,尾 tile 至多一个),仅 store mask;`_select_config` 携带离线 sweep 选出的 tile 配置。
3. **row-major 权重布局**:量化权重存 `[K, N]` 且 contiguous(而非惯用的 `[N, K]` 转置视图),避免 BLOCK_N 切片按 K 步长逐字节读;每 GEMM 1.05–1.33x、端到端 1.06x,单测钉住该布局防静默回归。
4. **护栏**:`gfx1250_fp8_scaled_mm_supported` 拒绝 tile 覆盖不了的 shape;`SGLANG_DISABLE_GFX1250_FP8_GEMM` 诊断开关;**BLOCK_SIZE_K 刻意封顶 64**——gfx1250 + triton 3.8.0 上 FP8 `tl.dot` 在 `BLOCK_SIZE_K >= 128` 时 miscompile(静默返回 NaN 且错误内核反而快约 40%,无正确性门的 autotune 会选出幻影 1730 TFLOPS);`triton_scaled_mm` 自身启发式会选 (128,128,128),修复前不得把 gfx1250 FP8 路由过去。

## 关键改动

```text
python/sglang/kernels/ops/gemm/fp8_gfx1250.py: 新增 A8W8 FP8 scaled_mm
  Triton tile(modulo 索引、BLOCK_K<=64 配置表)与 per-token 激活量化 kernel
python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py:
  use_gfx1250_triton_fp8 一次性判定;per-channel 权重量化分支;apply 先试
  _apply_gfx1250_triton_fp8,None 回落既有路径;权重存 row-major [K,N]
python/sglang/srt/environ.py: SGLANG_DISABLE_GFX1250_FP8_GEMM 诊断开关
test/registered/amd/test_fp8_gemm_gfx1250.py: tile 正确性 + dispatch seam 测试
```

关键片段(diff 摘录,modulo 环绕索引):

```python
offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
...
acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
```

## 性能声明

PR 正文,未独立核实。FLUX.2-dev 720p,`--enable-torch-compile true`,aiter attention,offload 全关:

- **MI450 dashboard 端到端**(同 harness/镜像/当日,DPX TP1 (A0)):BF16 基线 Total 51.787s → +PR+`--quantization fp8` **39.044s**;Denoise 48.7215→36.0457s(**−26.0%,1.35x**),Total **−24.6%(1.33x)**;text encode 与 VAE decode 不变。
- **单 gfx1250 隔离 A/B**(100 denoise steps,seed 42,各 2 次):BF16 0.7561 s/step → FP8 0.5573(**−26.3%,1.357x**);e2e 87.85→64.29s(−26.8%,1.367x);重复间波动 <0.5%。
- **row-major 布局再 −5.3%**(0.5537→0.5244 s/step,各 3 次);对 BF16 累计 **1.44x**(0.7561→0.5244 s/step)。profile 归因:三步 −174ms GPU 时间来自 GEMM,总 wall-clock −108ms,GPU-busy 1791.5→1683.6ms。
- 内核级参考(module docstring,MI 级 gfx1250 样机 2.4GHz):`torch.mm` BF16 ~70 TFLOPS、`torch._scaled_mm` ~500 TFLOPS、Triton tile BF16 ~770/FP8 ~1200 TFLOPS。
- 作者注明:dashboard 的 Denoise 列含 GPU tail drain,同一次运行比 per-step 高约 20%,两表不可跨比。

## 验证与回归风险

- 无 gsm8k 类精度 harness(diffusion 图像模型),验证分三层:轨迹发散性(逐步 denoise latent 对比,mean bias 每步均在 ±1e-3 内且变号);CLIP 分数(2 prompt,mean 30.49→30.44,delta −0.05,均在 ±0.5 内);PSNR/SSIM/LPIPS(18.55/0.8472/0.0901,仅报告非门槛,n=2 单 seed)。
- 新增注册 AMD 测试 `test_fp8_gemm_gfx1250.py`(register_amd_ci,stage-b-test-1-gpu-small-amd):tile 对 dequant 参考的正确性(FLUX2 720p shapes + ragged/small)、不支持 shape 拒绝、全零行量化、dispatch seam(row-major 钉死、kill switch)。
- 影响面:仅 `use_gfx1250_triton_fp8` 为真时生效(非 block quant、非 FP8 序列化、gfx1250);MI300X/MI355X/NVIDIA 不受影响(`is_gfx1250_supported()` 为 False 或已有 CUTLASS per-channel 路径)。
- 已知风险:triton 3.8.0 修复前 BLOCK_K>=128 miscompile 是目标平台固有地雷;kill switch 是诊断开关而非支持的退路(真要停 FP8 应去掉 `--quantization fp8`);Checklist 各项未勾选。

## 关联

同分类量化方向可参考同 batch 的 #46043(diffusion 启动优化);pack 未给出其他直接关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
