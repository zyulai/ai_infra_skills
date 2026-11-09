# PR #24757: [Perf] Use NVIDIA hardware-accelerated instruction for float to fp8_e4m3 quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2025-09-13 |
| 分类 | Quantization |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/24757 |

## 一句话结论

float→FP8_e4m3 量化改用 NVIDIA `__nv_cvt_float_to_fp8` 硬件指令:per-op 快 30–50%,Llama-3.3-70B-FP8 E2E 吞吐 +3%。

## 动机

vLLM CUDA 路径的 `scaled_fp8_conversion` 在 clamp 之后用 `static_cast<fp8_type>` 完成 float→FP8 转换,走软件模拟;NVIDIA GPU(sm80+)本有硬件 cvt 指令,而 ROCm 分支早已用硬件转换(`fp8::cvt_c10`),NVIDIA 侧一直未接入。FP8 模型每个 quant kernel(query/RMSNorm/SiluMul 等)都要反复做这个转换,软件路径是纯浪费。

## 优化点

`csrc/quantization/fp8/common.cuh` 的非 ROCm 分支由 `static_cast` 改调 `fp8::vec_conversion<fp8_type, float>(r)`;在 `nvidia/quant_utils.cuh` 为 `<c10::Float8_e4m3fn, float>` 特化该模板:`__CUDA_ARCH__ >= 800` 时用 `__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type)` 硬件指令(SATFINITE 饱和语义),更早架构回退 `static_cast`。通用 `vec_conversion` 模板补上默认 `__NV_E4M3` 参数后保持原返回 x 的行为,其余转换仍被 `#if 0` 屏蔽以控制二进制体积。前置的 clamp(`fmaxf/fminf` 到 quant_type_max)保持不变,只替换最终 cast 步骤。

## 关键改动

```text
csrc/quantization/fp8/common.cuh: scaled_fp8_conversion NVIDIA 分支改走 vec_conversion
csrc/quantization/fp8/nvidia/quant_utils.cuh: 新增 float→Float8_e4m3fn 硬件指令特化
```

```cpp
// float -> c10::Float8_e4m3fn
template <>
__inline__ __device__ c10::Float8_e4m3fn
vec_conversion<c10::Float8_e4m3fn, float>(
    const float& a, const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return static_cast<c10::Float8_e4m3fn>(a);
    #else
  return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type),
                            c10::Float8_e4m3fn::from_bits());
    #endif
}
```

## 性能声明

- Per-op(约 30–50% improvement):main vs PR:Attn Query Quant 3.488 → 2.464 µs,RMS Norm Quant 3.392 → 2.336 µs,SiluMul Quant 8.288 → 4.160 µs。PR 未注明测量硬件与方法细节。
- E2E:nvidia/Llama-3.3-70B-Instruct-FP8,`kv_cache_dtype=fp8`,TP1:output token throughput 4073.02 → 4182.03 tok/s,Median TPOT 29.65 → 28.91 ms(约 3% E2E improvement)。
- 精度:GSM8K 5-shot,limit 500,temperature 0:flexible-extract 0.940 / strict-match 0.904。

## 验证与回归风险

`tests/kernels/quantization/test_fp8_quant.py` 109 passed;E2E GSM8K 通过。风险:仅覆盖 NVIDIA 上 `fp8_type = c10::Float8_e4m3fn` 的转换路径(模板仅此特化,其余类型仍走原路径);sm80 以下架构回退软件 cast;硬件指令使用 `__NV_SATFINITE` 饱和语义,与原 `static_cast` 在边界值上的行为差异 PR 未讨论;`vec_conversion` 通用签名默认参数变更可能影响同头文件其他调用点(素材 diff 中其余仍被 `#if 0` 屏蔽)。

## 关联

素材未提及具体前序/后续 PR。索引同分类另有 pr-12978(blockwise CUTLASS StreamK)、pr-18844(scaled FP8 quant 调参)、pr-32520(FP4 量化 kernel 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
