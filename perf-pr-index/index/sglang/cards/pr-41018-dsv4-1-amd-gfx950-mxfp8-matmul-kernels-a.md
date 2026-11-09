# PR #41018: dsv4.1-amd: gfx950 MXFP8 matmul kernels and fp8-grid producers

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kevin-mii |
| 合并 | 2026-09-25 |
| 分类 | Quantization(major) |
| Label | quant, amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 因超长丢弃 6 个文件,仅保留 tuning 脚本、mxfp8_gemv_gfx95.cuh、silu_and_mul_clamp_hip.py、gfx95_batched_gemm_bf16_fp8_grid.py 等 |
| 链接 | https://github.com/sgl-project/sglang/pull/41018 |

## 一句话结论

DeepSeek-V4.1 AMD 系列 kernel PR(stack 1/4):为 gfx950 建 32x32-block ue8m0 MXFP8 原生 GEMV/GEMM 路线与 fp8-grid 融合 producer,tuned tile 较默认最高快 2.2x。

## 动机

gfx950 上 32x32-block ue8m0 fp8 checkpoint(如 DeepSeek-V4.1-Flash)此前缺少原生 MXFP8 dense 计算路线。该 PR 是 dsv4.1 AMD 系列的第一个 kernel PR:kernel 先落 main 但不接调用方,模型调用点由本 stack 的 integration PR 接入(沿用 CUDA dsv4.1 的组织方式 #39646/#39652/#39653/#39656/#39664/#38798)。

## 优化点

(1) 原生 MXFP8 dense 路线:`mxfp8_native_amd_gfx95.py` + `mxfp8_gemv_gfx95.cuh`——权重在 load 时一次性重排进 scaled-MFMA lane order 并保持 fp8,K 补零到 128 的倍数、activation 尾部 masked;M<=32 走基于 `v_mfma_scale_f32_16x16x128_f8f6f4` 的 HIP GEMV(fp32 累加,固定求和顺序,重复调用 bitwise 相等),更大 M 走 Triton `tl.dot_scaled` GEMM。输入可以是 bf16、位于 fp8 grid 上的 bf16、或 fp8 + ue8m0 scales,三种编码给出相同比特位(bf16 输入在寄存器内按 ue8m0 规则量化:scale 为 >= amax/448 的最小 2 的幂,RNE e4m3,先 clamp ±448)。(2) tile 表:在 DeepSeek-V4.1-Flash TP4 形状上按 M bucket 调参, HIP-graph replay 轮转超过 LLC 的权重副本模拟 decode 时的 HBM 流式读取。(3) fp8-grid helpers(`fake_quant_fp8_activation`/`fp8_grid_quantize`,与 CUDA `per_token_group_quant` 同 ue8m0 规则)及 bf16-dequant 线性兜底、`Fp8GridActivation`/`Mxfp8Activation` wrapper 让下游跳过自己的 quant。(4) 融合 producer:RMSNorm(+residual)+quant、clamped SiLU-mul+quant、wo_a batched GEMM epilogue 内做 quant(固定顺序 split-K,最多 64 行)。

## 关键改动

```text
python/sglang/kernels/ops/quantization/mxfp8_native_amd_gfx95.py: 原生 MXFP8 dense 路线
  (prepare_mxfp8_native_weight / GEMV / dot_scaled GEMM)
python/sglang/kernels/jit/csrc/deepseek_v4/mxfp8_gemv_gfx95.cuh: gfx950-only HIP GEMV
  kernel(仅 gfx950 可编译,其他 target 直接 #error)
python/sglang/kernels/ops/quantization/mxfp8_gemv_gfx95_configs.json: 按 M bucket 的
  tuned tile 表
python/sglang/kernels/ops/quantization/mxfp8_amd_gfx95.py: 追加 fp8-grid helpers 与
  wrapper(既有函数与调用方不变)
python/sglang/kernels/ops/activation/rmsnorm_fake_quant_amd_gfx95.py: RMSNorm+quant 融合
python/sglang/kernels/ops/activation/silu_and_mul_clamp_hip.py: clamped SiLU-mul 单 launch,
  可选 fp8-grid/fp8 输出
python/sglang/kernels/ops/gemm/gfx95_batched_gemm_bf16_fp8_grid.py: wo_a batched GEMM,
  epilogue 内 fp8-grid rounding(>64 行与 aiter tile loop bitwise 一致)
benchmark/kernels/quantization/tuning_mxfp8_native_gfx95.py: tile 调参脚本
```

关键摘录(GEMV MFMA 调用):

```cuda
acc[t][b] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    av, bv, acc[t][b], 0, 0, 0, sa, 0, sb);
```

## 性能声明

MI355X(ROCm 10.0.0, torch 2.11, triton 3.8, aiter 按 Dockerfile pin):tuned tiles 较默认配置最高 **2.2x faster**(未注明具体形状);原生 K tail 路线在 (5120, 576) 形状、M 至 4096 时耗时为 bf16-dequant 路线的 0.63–0.93x。PR 明确注明:本分支未单独执行端到端 serving 与 accuracy 测试(由 integration PR 覆盖)。

## 验证与回归风险

`test_mxfp8_amd_gfx95.py` 与 `test_fp8_grid_producers_gfx95.py` 通过:fp64 参考值比对、三种输入编码 bitwise 一致(含非有限激活)、与 aiter `batched_gemm_bf16` bitwise 匹配;既有 `test_minimax_m3_mxfp8.py` 亦通过。回归风险低:`mxfp8_amd_gfx95.py` 只追加代码,其余文件全部为新增且 HIP-only,`.cuh` 对非 gfx950 target 拒绝编译。CI 状态显示 base/extra/AMD ROCm 10 测试均为失败状态(pack 中记录,原因未在正文说明)。

## 关联

同 stack 后续:#41019、#41020、#41021(dsv4.1 AMD 系列 stack 2/4–4/4,integration PR 在其中);对齐 CUDA 侧布局:#39646、#39652、#39653、#39656、#39664、#38798。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
