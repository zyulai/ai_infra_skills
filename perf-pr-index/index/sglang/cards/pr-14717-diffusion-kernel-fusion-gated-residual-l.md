# PR #14717: [diffusion] kernel fusion: gated residual layernorm scale shift and layernorm scale shift kernel fusion for Qwen-Image, WAN and HunyuanVideo

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jianyingzhu |
| 合并 | 2026-02-04 |
| 分类 | Attention / Kernel(major) |
| Label | sgl-kernel, run-ci, diffusion |
| Diff | diffstat 缺失(素材未提供统计);diff 展示 5 个文件,另有 5 个文件因篇幅被裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/14717 |

## 一句话结论

为扩散模型把 residual/gating/LayerNorm/scale-shift 融合成单个 kernel,Qwen-Image E2E -11.1%、Wan2.1-1.3B -14.0%。

## 动机

Profiling 显示 Qwen-Image、WAN、HunyuanVideo 的 DiT pipeline 中存在大量 GPU bubble,来自调制链路上多个小 kernel 的 launch 开销:每次 AdaLN 调制需要 norm、乘 scale、加 shift,门控残差路径还要 gate*x、+residual,各为独立算子。将它们融合成单 kernel 可消除这些 bubble 与 launch 开销。

## 优化点

两个融合 kernel:(1) `fused_scale_residual_layernorm_scale_shift`,一次完成 `residual_out = residual + gate * x` 与 `y = norm(residual_out) * (1 + scale) + shift`;(2) `fused_layernorm_scale_shift`,完成 `y = norm(x) * (1 + scale) + shift`。实现为 CuTe DSL(CUTLASS)JIT kernel:每 CTA 处理一个 (batch, seq) 行,fp32 累加做 mean/var 归约(warp + CTA 两级 reduce),128-bit 向量化拷贝(LDG.128);支持 LayerNorm 与 RMSNorm、affine 可选;scale/shift/gate 支持 [1]、[D]、[B,D]、[B,1,D]、[B,S,D] 与 4D 帧 [B,F,1,D](BSFD 索引)等广播形态;要求 D 为 256 的倍数且 ≤8192 以保证无谓词向量加载;编译缓存按 (norm_type, dtype, ndim, D) 特化。`multimodal_gen` 的 layernorm 层在 kernel 可用时切换到融合路径。

## 关键改动

```text
python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py: 两个融合 kernel 的 CuTe DSL 实现 + 编译缓存
python/sglang/jit_kernel/diffusion/cutedsl/common/norm_fusion.py: layernorm/rmsnorm CTA 级归约与 BSFD 广播索引
python/sglang/jit_kernel/diffusion/cutedsl/common/reduce.py: warp/CTA reduce_sum
python/sglang/jit_kernel/tests/test_fused_norm_scale_shift.py: 正确性测试(3 dtype × 多 shape × 多广播模式)
python/sglang/jit_kernel/benchmark/bench_fused_norm_scale_shift.py: microbench(native vs CuTeDSL vs CUDA C)
```

```python
# For scale_residual_norm_scale_shift, output:
# - residual_out = residual + gate * x
# - y = norm(residual_out, weight, bias) * (1 + scale) + shift
value = tXrX.load()
if cutlass.const_expr(isinstance(tGrG, cute.Tensor)):
    value = tGrG.load() * value
if cutlass.const_expr(isinstance(tRrR, cute.Tensor)):
    value = value + tRrR.load()
```

## 性能声明

Microbench(B=1,单位 µs,SGLang Native → CUDA C 融合版,GPU 型号 PR 未注明;完整表见 PR):如 S=4096/D=4096 layer+affine 225.408 → 52.544;S=128/D=1024 layer+affine 31.68 → 7.584;rms 变体从 ~12-81 µs 降至 ~7.3-52 µs。

E2E(sglang generate,各模型配置见 PR 命令,部分启用 CacheDiT 与 torch compile):

| 模型 | E2E Latency 基线 → 本 PR | 变化 |
| --- | --- | ---: |
| Qwen-Image-2512(1024×1024, 50 步) | 7072.14 → 6289.26 ms | -11.1% |
| Wan2.2-T2V-A14B(720p, 4 GPU) | 362312.77 → 357770.03 ms | -1.3% |
| Wan2.1-T2V-1.3B | 71606.89 → 61612.73 ms | -14.0% |
| HunyuanVideo(480p, 65 帧) | 71280.41 → 70908.72 ms | -0.5% |

## 验证与回归风险

新增 `test_fused_norm_scale_shift.py` 覆盖 fp16/bf16/fp32、layer/rms、affine 有无、9 种 scale/shift/gate 广播模式,对 torch 参考实现 assert_close(fp32 tol 1e-5,其余 5e-2),并覆盖 Hunyuan/Wan/Qwen 的真实 shape(如 [1,115200,3072])。PR 声明与现有 PyTorch/Triton 实现数值 parity。回归风险:融合路径有硬约束(D 为 256 倍数且 ≤8192,不满足抛错);Wan2.2 的 DecodingStage 在该轮测量中 +47.9%(8172→12084 ms),作者未在正文解释;`torch._dynamo.disable` 使这些函数脱离 torch compile 追踪。

## 关联

合作者 @AichenF、@yingluosanqian。同区域其他 PR 编号素材未提供,写无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
