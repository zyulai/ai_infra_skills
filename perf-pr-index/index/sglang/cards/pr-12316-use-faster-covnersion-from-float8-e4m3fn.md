# PR #12316: use faster covnersion from float8_e4m3fn to bfloat16

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mingfeima |
| 合并 | 2025-12-04 |
| 分类 | Quantization |
| Label | sgl-kernel, intel, cpu, run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/12316 |

## 一句话结论

CPU FP8 GEMM 的 fp8→bf16 转换指令数从约 25 条降到 6 条,平均提速 11%。

## 动机

CPU 上的 float8 GEMM(tinygemm 路径)在对 B 矩阵做 `unpack_B` 时需要把 float8_e4m3fn 反量化为 bf16,旧的 `CVT_FP8_TO_BF16` 实现每元素约需 25 条指令,是 FP8 GEMM 性能热点。作者参考 Zhihu 上的位运算技巧与 chitu 项目 CUDA 端 marlin dequant 实现,重写为更快的转换逻辑。

## 优化点

在 `vec.h` 新增 AVX-512 向量化函数 `CVT_FP8_TO_BF16_EXT`:先用 `_mm512_cvtepu8_epi16` 把 fp8 字节扩展为 16-bit,再用掩码拆出符号位(0x80,左移 8 位对齐 bf16 符号位)与指数+尾数(0x7F,左移 4 位),最后用单条 `_mm512_ternarylogic_epi32`(立即数 0b11111110)把符号、0x4000 偏置与指数尾数三路合并成 bf16,整个转换收束到约 6 条指令。同时引入 `kFP8_BIAS`(0x3b800000,即 fp32 的 1/256)预先乘进 scale 向量,把 fp8 无偏指数表示与 bf16 的差异折算到 block scale 里,替代旧的逐元素处理路径。`gemm_fp8.cpp` 中 `unpack_B` 与 `tinygemm_kernel_nn` 的两处 B 矩阵转换点均切换到新函数。

## 关键改动

```text
sgl-kernel/csrc/cpu/vec.h: 新增 CVT_FP8_TO_BF16_EXT 位运算转换与 kFP8_BIAS 宏
sgl-kernel/csrc/cpu/gemm_fp8.cpp: unpack_B 与 tinygemm_kernel_nn 切换到新转换,scale 预乘 bias
```

关键片段:

```cpp
inline __attribute__((always_inline)) __m512bh CVT_FP8_TO_BF16_EXT(__m256i a) {
  const __m512i mask0 = _mm512_set1_epi16(0x80);  // sign bit
  const __m512i mask1 = _mm512_set1_epi16(0x7F);  // exponent and mantissa
  const __m512i mask2 = _mm512_set1_epi16(0x4000);
  __m512i x = _mm512_cvtepu8_epi16(a);
  __m512i vsign = _mm512_and_si512(x, mask0);
  vsign = _mm512_slli_epi16(vsign, 8);
  __m512i vexp_and_mant = _mm512_and_si512(x, mask1);
  vexp_and_mant = _mm512_slli_epi16(vexp_and_mant, 4);
  return (__m512bh)(_mm512_ternarylogic_epi32(vsign, mask2, vexp_and_mant, 0b11111110));
}
```

## 性能声明

PR 正文: "on average 11% performance improvements",基准脚本为作者仓库 mingfeima/sgl-cpu-tests 的 bench_gemm.py。PR 未注明具体测试条件(硬件型号、GEMM 规模、模型均未给出,仅能从 label 推断面向 Intel CPU)。动机中提到旧实现约 25 条指令、新实现 6 条指令。

## 验证与回归风险

精度由 sglang 已有的 fp8 gemm 与 moe 测试覆盖(PR 正文声明),无新增测试文件(diff 中未见)。风险集中在 CPU FP8 GEMM 路径:`unpack_B` 与 `tinygemm_kernel_nn` 是所有走 sgl-kernel CPU FP8 反量化推理的模型入口,转换语义若与旧实现有偏差会影响所有 FP8 W8A16 CPU 推理的数值; ternarylogic 合并中 0x4000 常数对应 fp8→bf16 的指数偏置,对 NaN/Inf 等特殊值的处理与旧实现是否一致 PR 未注明。不影响 GPU 路径。

## 关联

同分类(Quantization):PR #7278(CUTLASS FP8 Blockscale MoE,GPU 侧 FP8 内核)、PR #42957(TileLang FP8 KV cache)。上游参考:chitu marlin dequant.h。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
