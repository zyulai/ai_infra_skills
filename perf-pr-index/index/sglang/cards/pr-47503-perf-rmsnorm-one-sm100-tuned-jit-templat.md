# PR #47503: [Perf] rmsnorm: One SM100-tuned JIT template for plain, fused residual-add and fp8 per-tensor quant

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-11-02 |
| 分类 | Quantization |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失;素材 diff 截去 12 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/47503 |

## 一句话结论

一个 SM100 调优的 JIT rmsnorm 模板统一 plain/fused-add/fp8 quant,全面快于 flashinfer CuTe-DSL(中位 0.84x)。

## 动机

CUDA 上 JIT rmsnorm 内核此前只用于 HF 语义路径,`RMSNorm.forward_cuda` 的 plain、fused-add 与 per-tensor FP8 变体跑 flashinfer。flashinfer CuTe-DSL 内核每行付两次 CTA barrier、三次 shared-memory pass 和一次 weight 重读,且在 `griddepcontrol.wait` 之前不发出任何访存;decode 尺寸下每次 launch 损失 0.3-0.8us(B200),hidden < 4096 时带宽只有 3.8-4.6 TB/s。

## 优化点

elementwise/rmsnorm.cuh 收敛为一个模板内核 + `RMSNormMode`(PLAIN/FUSED_ADD/QUANT/FUSED_ADD_QUANT)+ HF rounding flag:一行由 threads_per_row 个线程处理、rows_per_cta 行每 CTA,Blackwell 32B 向量(其他 arch 16B),行数据驻留寄存器,weight 在 griddepcontrol.wait 前预载,跨 warp 的行只需一次 barrier 归约,短行(32-496)取一个 warp 的 2 的幂切片;Blackwell 用 `fma.rn.f32.bf16` 混合 FMA(fused residual 用一条 `add.rn.bf16x2`),fp8 输出用 `cvt.rn.satfinite.e4m3x2.f32` 打包成 16B store;删除 fused_add_rmsnorm.cuh 与 rmsnorm_hf.cuh。schedule (vec_bytes, threads_per_row, rows_per_cta) 由 Python 按测量表选择、同一签名的两套 schedule 一次性 parallel_compile,调用时按 token 数选择;`RMSNorm.forward_cuda` 与 `forward_with_per_tensor_quant_fusion` 在 SM100 bf16/fp16 改用该 JIT 家族,其他 arch 保持 flashinfer;HF 路径维持所有 CUDA arch 使用。

## 关键改动

```text
python/sglang/kernels/jit/csrc/elementwise/rmsnorm.cuh: 统一模板内核(RMSNormMode + schedule 模板参数),替换 fused_add_rmsnorm.cuh/rmsnorm_hf.cuh
python/sglang/kernels/ops/layernorm/norm.py: schedule 表选择与 public 入口(diff 未保留该文件)
python/sglang/srt/layers/layernorm.py: SM100 上 bf16/fp16 静态 gate 到 JIT 家族
```

```cpp
enum class RMSNormMode {
  PLAIN,            // out = norm(x) * w
  FUSED_ADD,        // res = T(x + res); x = norm(res) * w
  QUANT,            // out8 = fp8(norm(x) * w / scale)
  FUSED_ADD_QUANT,  // both
};
using enum RMSNormMode;
```

## 性能声明

B200,bf16,CUDA graphs,median us/launch,cold L2(括号内为 flashinfer CuTe-DSL):rmsnorm D=4096:N=1 1.18(1.54)、N=2048 7.72(8.26)、N=8192 22.3(23.8);D=3072 N=2048 5.69(8.46);fused_add D=8192 N=2048 22.8(24.1);rmsnorm_quant D=4096 N=2048 6.15(7.34);fused_add_quant D=8192 N=2048 20.5(22.2)。16 hidden sizes × 7 token counts:plain 112/112 更快(中位 0.84x),fused 112/112,quant 28/28(0.83x),fused quant 28/28(0.89x);短行(64-384)在 N=8192 快 25-40%;producer→norm→consumer PDL 链中 D=4096、N=1 的 exposed increment 从 1.63 降到 1.05us。

## 验证与回归风险

test_rmsnorm.py 重写(plain/fused/HF/quant/strided/empty,含并行 prebuild fixture),B200 全量 2777 passed;test_fused_add_rmsnorm.py 与 test_rmsnorm_hf.py 折入。精度:plain 与 flashinfer 差 <1 bf16 ulp(<2e-4 元素,累加顺序所致),fused residual 对 `(x.float()+res.float()).to(bf16)` bit-exact,HF 输出对 fp32 参考 bit-exact,fp8 除 ≤2e-6 元素外与 flashinfer bits 相等。影响面:SM100 上 bf16/fp16 的 RMSNorm CUDA 路径切换为 JIT;其他 arch 与 HF 路径不变;fp8 quant 模式 CUDA-only(ROCM static_assert 拒绝)。

## 关联

无(素材未列出同区域其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
