# PR #4104: Add awq dequantize kernel to sgl with 1x to 3x speedup

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2025-03-12 |
| 分类 | Quantization(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 7 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/4104 |

## 一句话结论

为 sgl-kernel 新增 AWQ 反量化 CUDA kernel:去掉输出赋值处的拷贝与 for 循环,对 vllm CUDA 实现提速 1x–4x(标题称 1x–3x)。

## 动机

对应需求 #2965。vllm 的 `awq_dequantize` CUDA 实现在输出赋值的末尾存在额外的 copy 与 for 循环(vllm `gemm_kernels.cu` L403 附近),是主要开销来源。本 PR 把该 kernel 改编进 sgl-kernel,消除这段冗余工作,使 AWQ 权重加载/预处理路径更快。

## 优化点

新增 CUDA kernel `awq_dequantize`(qweight int32、scales/qzeros → fp16 输出)。核心是 `dequantize_s4_to_fp16x2`:用 `lop3.b32` 位逻辑指令配合 magic number(0x64006400)与掩码,把打包的 4-bit 整数直接重排成 fp16x2,再用 `sub.f16x2`/`fma.rn.f16x2` 完成偏置与缩放——整个序列只需 1 次移位(利用寄存器打包格式与无符号整数的约定,elt_23/elt_67 无需先移位)。`dequantize_weights` kernel 中每个线程取一个 uint4(8 个 fp16 权重),减 zeros、乘 scale 后以单次 `*(uint4*)` 写入输出,替代 vllm 末尾的逐元素 copy + for loop;host 侧按 16x16 线程块划分网格,输出以 scales 的 dtype 分配。操作注册进 torch library(`sgl_kernel::awq_dequantize`,CUDA impl)并暴露 Python API;要求 `__CUDA_ARCH__ >= 750`。

## 关键改动

```text
sgl-kernel/csrc/gemm/awq_kernel.cu: AWQ 4-bit→fp16 反量化 CUDA kernel(lop3/f16x2 指令、uint4 直写输出)
sgl-kernel/csrc/torch_extension.cc 与 include/sgl_kernel_ops.h: 注册 sgl_kernel::awq_dequantize
sgl-kernel/python/sgl_kernel/gemm.py: Python 绑定
sgl-kernel/tests/test_awq_dequant.py: 与 vllm 实现对齐的参数化测试
sgl-kernel/benchmark/bench_awq_dequant.py: 42 形状性能对比脚本
```

```cpp
uint4 weight_fp16 = dequantize_s4_to_fp16x2(qweight[col + row * qweight_cols]);

asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(zeros.x));
asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(weight_fp16.x) : "r"(weight_fp16.x), "r"(loaded_scale.x));
// (.y/.z/.w 同样 sub+mul,摘录省略)

half* output_ptr = output + 8 * col + 8 * row * qweight_cols;
*(uint4*)output_ptr = weight_fp16;
```

## 性能声明

与 vllm CUDA `awq_dequantize` 对比(`triton.testing.do_bench`,42 个 (qweight_row, qweight_col) 形状,fp16):大矩阵收益最大——18944×4736 为 3465.47→897.12 us(约 3.9x)、3584×4736 为 669.73→182.53 us(约 3.7x)、18944×448 为 356.13→99.23 us(约 3.6x);小形状基本持平(如 128×16:12.99→12.80 us)。正文总结:"Achieved 1x - 4x speed up compared to vllm awq_dequantize cuda implementation"(PR 标题为 1x to 3x)。CUDA 对照未注明硬件;与 vllm Triton 实现的对照注明 H100,结果略优、基本持平(如 18944×4736:898.58→897.84 us)。accuracy:42 形状下与 vllm 输出 `assert_close`(rtol 1e-3/atol 1e-5),bench 打印 "All implementations match"。

## 验证与回归风险

新增 `sgl-kernel/tests/test_awq_dequant.py`(42 形状参数化,与 vllm `_custom_ops.awq_dequantize` 对齐);benchmark 脚本含 `calculate_diff` 一致性检查。约束:kernel 仅支持 SM75+(`__CUDA_ARCH__ >= 750`,否则 assert);网格按 qweight 维度被 16 整除划分,group_size 由 `qweight_rows / scales_rows` 推断,素材未写非整除形状的处理。checklist 各项均未勾选;pack 未包含 CI States 记录。

## 关联

动机引用 #2965(前序需求);kernel 改编自 vllm awq_dequant CUDA 实现;作者提及后续可移植 vllm 的 AWQ Triton 实现(未在本 PR 内)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
