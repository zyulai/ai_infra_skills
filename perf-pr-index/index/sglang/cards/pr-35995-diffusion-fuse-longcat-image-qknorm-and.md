# PR #35995: [diffusion] Fuse LongCat-Image QKNorm and interleaved RoPE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-24 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci, diffusion, jit-kernel, bypass-fastfail, run-ci-extra, mergeable |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/35995 |

## 一句话结论
把 LongCat-Image 的 QKNorm 与全宽 interleaved RoPE 融合进共享 JIT kernel,e2e 延迟降约 17%,kernel 级 13.61x。

## 动机
LongCat-Image 此前用 Diffusers 链:QK-norm 走 SGLang fused RMSNorm,RoPE 因 `axes_dims_rope=[16,56,56]` 之和等于 head_dim=128(全旋转)与 flashinfer cos_sin_cache 格式(rotary_dim ≤ head_dim)不兼容,只能单独用 `apply_rotary_emb`。共享 JIT QKNorm+RoPE kernel 此前不支持全宽 interleaved FP32 cos/sin cache;基线 trace 中 split 链路约 2.0s,含 4k 次 QKNorm 调用、18k 次 copies、12k 次 muls、6k 次 adds、12k 次 cats、6k 次 negs。

## 优化点
扩展共享 JIT `fused_qknorm_rope_warp` kernel 支持全宽 interleaved FP32 cos/sin cache:新增 `rotary_add_fp32`/`rotary_sub_fp32`(CUDA 上用 `__fmul_rn`/`__fadd_rn`),在 `cache_has_full_width` 时按 lane 计算成对 cache 下标,并放宽 static_assert 使 `round_norm_before_rope` 在 FP32 cache 下也允许与激活 dtype 不匹配;Python 门控 `_can_use_fused_qknorm_rope` 相应放开。LongCat-Image 的 joint/single attention block 改走 `apply_qk_norm_rope` 融合路径(模型侧用 `BitExactFusionGate` 在运行时做一次与 Diffusers 链的 bit-exact 校验,不匹配或 JIT 失败即回退);benchmark 增加 `longcat_1024` 形状。

## 关键改动

```text
python/sglang/kernels/jit/csrc/diffusion/qknorm_rope.cuh: 全宽 interleaved FP32 cache 的 RoPE 路径(rotary_add/sub_fp32)
python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py: _apply_longcat_qknorm_rope 融合入口 + BitExactFusionGate 运行时校验回退
python/sglang/kernels/ops/diffusion/rope/qknorm_rope_jit.py: 门控条件放宽(cache_dtype 可为 fp32)
test/registered/kernels/benchmark/diffusion/bench_qknorm_rope.py: 新增 longcat_1024 用例
```

## 性能声明
1x H200、`meituan-longcat/LongCat-Image`、1024x1024、50 steps、seed 42,ABBA 运行、基线 `3c69a4c7`:eager 端到端 lossless 8.9970/9.1171s → 7.4039/7.5243s(-17.47%),high -17.48%;Breakable CUDA Graph lossless e2e -16.88%,high denoise stage 8.9729→7.4392s(-17.09%)。kernel microbench `longcat_1024`:split 741.28us vs fused 54.46us,13.61x;PR trace 177.44ms/4k 次融合调用,总 GPU 时间约 8.74s → 7.20s。输出与基线 byte-identical(lossless/high 均给出 SHA256)。

## 验证与回归风险
`test_rope.py` + `test_model_fast_paths.py` 1298 passed(含新增全宽 interleaved cache 的 fp16/bf16 bit-exact 用例与 LongCat 模型 fast-path 测试);targeted runtime-gate model test passed;最终 lossless/high BCG smoke capture/replay 成功、输出 hash 精确、无回退。风险:融合路径带运行时 bit-exact 门控,不匹配平台会静默回退 Diffusers 链(此时无收益);KV packing 不支持全宽 cache(门控排除);AMD ROCm 7.2 CI 为 pending 标记。

## 关联
同区域其他 diffusion 融合见 `python/sglang/kernels/ops/diffusion/README.md` 的 kernel 契约表(fused_inplace_qknorm_rope 等);同分类其他 PR 编号素材未给出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
