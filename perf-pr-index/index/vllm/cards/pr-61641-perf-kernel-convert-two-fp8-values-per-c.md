# PR #61641: [Perf][Kernel] Convert two FP8 values per cvt in the FP8 quant kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-11-02 |
| 分类 | Quantization(major) |
| Label | ready, nvidia, quantization |
| Diff | 素材未提供 diffstat 数字;pack diff 收录全部核心 CUDA 头/源文件与测试改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/61641 |

## 一句话结论

sm_100+ 上 FP8 量化 kernel 改用 `__nv_cvt_float2_to_fp8x2` 一次转换两个值、由饱和 cvt 取代显式 clamp,fused RMSNorm quant 提速 1.05–1.24x,输出 bit-identical。

## 动机

FP8 的 `cvt` 硬件指令本身支持一次转换两个值(#60927 已在 per-token-group packed 路径启用),但 vLLM 其余 FP8 量化路径的标量 helper `scaled_fp8_conversion` 每次只转一个值(以零为 partner 发射指令),还配了一条显式 clamp。在 sm_100+ 上这是双重的指令浪费,而 FP8 量化(RMSNorm+quant、scaled_fp8_quant、merge_attn_states)遍布每次前向。

## 优化点

新增向量 helper `scaled_fp8_conversion_vec<is_scale_inverted, fp8_type, N>`:在架构 gate `VLLM_FP8_CVT_X2`(`VLLM_FP8_CVT_X2_MIN_ARCH 1000`,即 sm_100+)下把元素两两配对,用 `__nv_cvt_float2_to_fp8x2(..., __NV_SATFINITE, __NV_E4M3)` 一次转两个值;饱和转换语义取代了显式 clamp;奇数尾部与非目标架构回落到原 scalar 函数(其本身不变)。调用侧:`common.cu` 的三种 scaled_fp8_quant kernel 通过新 functor `ScaledFp8QuantOp`(同时提供标量与 16 宽两个 op)喂给 `vectorize_with_alignment`;fused RMSNorm quant kernel 经新的 `ScaledQuant::quant_vec_fn` 分发(int8 走普通循环)。由于跳过 clamp 后 `0 * inf` 会得到 FP8 NaN,所有喂反演 scale 的调用点改用 `fp8_reciprocal_scale`(零 scale 返回 0)。非 NaN 输入下输出与 main 逐位相同;唯一行为变化是 all-zero 输入的动态 per-tensor 量化:main 写出 448(+0 scale),本 PR 正确写出 0。

## 关键改动

```text
csrc/quantization/w8a8/fp8/common.cuh: 新增 scaled_fp8_conversion_vec(N 值两两 cvt)与 fp8_reciprocal_scale
csrc/quantization/w8a8/fp8/common.cu: 三种 scaled_fp8_quant kernel 经 ScaledFp8QuantOp functor 走向量路径
csrc/quantization/fused_kernels/quant_conversions.cuh / layernorm_utils.cuh: fused RMSNorm quant 经 quant_vec_fn 接入
csrc/attention/merge_attn_states.cu: FP8 输出合并改用向量转换
csrc/core/cuda_compat.h: VLLM_FP8_CVT_X2 架构 gate(>= sm_1000)
```

关键片段(common.cuh):

```cuda
#if VLLM_FP8_CVT_X2
  for (; i + 1 < N; i += 2) {
    const float2 x = is_scale_inverted
        ? make_float2(val[i]*scale, val[i+1]*scale) : ...;
    const __nv_fp8x2_storage_t q =
        __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
```

## 性能声明

B300,FlashInfer `bench_gpu_time_with_cupti`,CUDA graph + cold L2,取中位数,526 个用例(batch 1–8192,hidden 4096/5120/7168/2880/1030/1025/17,bf16/fp16/fp32,含非对齐 view),两轮交替运行取 min,run-to-run 波动约 ≤1%,总体 geomean 1.023x。bf16 按 hidden 取 geomean:
- `rms_norm_per_block_quant`(1x128):1.11x(1 行)→ 1.24x(8192 行);`rms_norm_dynamic_per_token_quant`:1.04–1.10x。
- static / dynamic per-tensor / per-token `scaled_fp8_quant`:1.00–1.08x;`merge_attn_states` FP8 输出最高 1.03x。
- 绝对时间例(bf16,hidden 4096):`rms_norm_per_block_quant` 8192 行 102.91→83.75 µs;`rms_norm_dynamic_per_token_quant` 2048 行 32.67→27.62 µs;static quant 8192 行 20.96→19.39 µs。
- 小 batch:bf16 用例在 ≤128 行的最差为 0.98x;hidden 1030/1025/17 基本持平(1.00x,走未改动的 scalar 前后缀路径)。已知回归:fp32 输入 hidden 1030 @2048 行的 static quant 慢 4.5%(9.54→9.98 µs,源码未改,系同 kernel 内相邻改动的 codegen 副作用)。

## 验证与回归风险

验证强度高:逐字节 dump 全部触及 op 的输出(FP8 字节与 scale)与 main 构建 build 对照,bf16/fp16/fp32、5 个 hidden、对齐/非对齐、各量化变体共 515 个输出张量 0 差异;遍历全部 2^32 个 fp32 位型验证 clamp-then-cvt 与裸 cvt 在非 NaN 输入上一致(唯一差异是 NaN 保持 FP8 NaN 而非变 +448);B300 上 5118(后 5803)个测试通过;新增 `test_dynamic_per_tensor_fp8_quant_all_zero` 在 main 上失败、本 PR 通过。风险面:指令从 sm_89 就存在但仅在 Blackwell 验证过,降 `VLLM_FP8_CVT_X2_MIN_ARCH` 需先在其他 GPU 验证;CI 的 L4/MI300 走不变路径;无 E2E serving benchmark —— 输出 bit-identical 故精度按构造不变;decode 规模 launch 上 fused RMSNorm quant 变动 ≤11%,其余 ≤4%。

## 关联

前序:#60927(per_token_group_quant 的 packed 路径,本 PR 把其 gate 统一到 `VLLM_FP8_CVT_X2`)。相关:#41115(per-token kernel launch 配置)。同批次量化 kernel 优化:#63101(MXFP4 激活量化)、#65791(SiLU-and-Mul per-block 量化向量化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
