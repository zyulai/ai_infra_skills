# PR #8248: [Kernel] [Triton] Memory optimization for awq_gemm and awq_dequantize, 2x throughput

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @rasmith |
| 合并 | 2024-09-06 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/8248 |

## 一句话结论

重写 AWQ Triton 内核的加载模式,awq_gemm/awq_dequantize 内存加载次数大减,端到端吞吐约 2 倍。

## 动机

AWQ 4-bit 权重以 int32 打包(每 word 装 8 个 4-bit 值)。原 awq_gemm 与 awq_dequantize Triton 内核为了展开打包权重,用 `tl.arange(0, BLOCK * 8) // 8` 这类索引,导致同一个 packed int32 被重复加载 8 次;scales/zeros 也按每个 group 内元素逐个重复加载。显存带宽被大量冗余读占用,成为 AWQ Triton 路径的瓶颈。

## 优化点

机制是把"重复加载 + 就地展开"换成"每值一次加载 + 寄存器内展开":权重侧只按 packed 维度加载一次(`tl.arange(0, BLOCK_SIZE_X)` / `BLOCK_SIZE_N // 8`),随后用三层 `tl.interleave` 在寄存器里把每个 int32 复制成 8 份,再做 shift/mask 反量化;scales 与 zeros 同样改为每个 group 只加载一次(`tl.arange(0, 1)` 的 group 级偏移),再用 `tl.broadcast_to` 扩到整个 tile。这样 qweight、scales、zeros 三路的全局内存读次数都降到原来的约 1/8,反量化逻辑(reverse AWQ order、shift & 0xF)不变。

## 关键改动

```text
vllm/model_executor/layers/quantization/awq_triton.py: awq_gemm_kernel 与 awq_dequantize_kernel 的加载/展开重写
```

```python
b = tl.load(b_ptrs, mask=masks_b)
b = tl.interleave(b, b)
b = tl.interleave(b, b)
b = tl.interleave(b, b)
...
zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))
```

## 性能声明

`benchmarks/benchmark_throughput.py --input-len 2048 --output-len 128 --quantization awq --model Qwen/Qwen2-7B-Instruct-AWQ`:优化前 Triton 6.14 req/s / 13351.06 tok/s;优化后 Triton 12.77 req/s / 27783.36 tok/s(约 2x);对比 AWQ CUDA 实现 11.51 req/s / 25049.29 tok/s,优化后 Triton 路径已超过 CUDA 版。PR 未注明测试硬件与精度条件。

## 验证与回归风险

PR 正文未列出新增/依赖的测试。风险点:interleave 展开顺序必须与 reverse AWQ order([0,4,1,5,2,6,3,7])严格一致,否则所有 AWQ 量化模型(Triton 反量化与 GEMM 路径,含 ROCM,因 label 含 rocm)输出错误;zero/scale 改为 group 级一次加载后,group_size 边界 mask 语义变化也需覆盖。后续若有 Marlin/Machete 等 AWQ 替代内核,行为可比对验证。

## 关联

同分类:AWQ 最初支持 #1032;AWQ/GPTQ 类低比特 GEMM 替代实现 #2497(Marlin)、#20268(Machete zp 支持)。
