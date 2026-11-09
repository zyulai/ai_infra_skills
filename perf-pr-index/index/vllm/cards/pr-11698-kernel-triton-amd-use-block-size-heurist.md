# PR #11698: [Kernel][Triton][AMD] Use block size heuristic for avg 2.8x speedup for int8 models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @rasmith |
| 合并 | 2025-01-08 |
| 分类 | Quantization |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,仅 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/11698 |

## 一句话结论

int8 模型的 triton_scaled_mm 弃用固定 32x32x32 tile,改用 SM90 int8 dispatch 启发式选块,平均提速 2.8x。

## 动机

compressed-tensors W8A8 int8 模型在 AMD GPU 上走 `triton_scaled_mm` Triton 内核,此前 block size 固定为 `(32, 32, 32)`,与矩阵形状无关。小 tile 对大 M/N 的 GEMM 意味着过少的每线程工作与过低的数据复用;CUTLASS 的 `scaled_mm_c3x_sm90_int8_dispatch.cuh`(第 116 行)已有一套按 M、N 分档的成熟启发式,可直接移植。

## 优化点

给 `triton_scaled_mm` 增加 `use_heuristic=True` 开关:按 `next_power_of_2(M)` 与 N 是否小于 8192 分四档选 `(BLOCK_M, BLOCK_N, BLOCK_K)`——M≤32 时小 N 取 (64,64,256)、大 N 取 (64,128,256);M≤64 取 (64,64,256);M≤128 取 (64,128,128);更大 M 取 (128,128,128)。即小 batch decode 用大 K 维加深流水,大 batch prefill 用大 M/N tile 提高计算密度。`block_size_sa/sb`(scale 的 block 大小)仍按原逻辑从所选 block size 派生。默认启用启发式,传 `use_heuristic=False` 可回退原固定值。

## 关键改动

```text
vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py: 增加按 M/N 分档的 tile 启发式
```

```python
if use_heuristic:
    is_small_N = N < 8192
    next_power_of_2_M = max(32, triton.next_power_of_2(M))
    if next_power_of_2_M <= 32:
        tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
    elif next_power_of_2_M <= 64:
        tile_shape = (64, 64, 256)
    elif next_power_of_2_M <= 128:
        tile_shape = (64, 128, 128)
    else:
        tile_shape = (128, 128, 128)
```

## 性能声明

(`benchmarks/benchmark_latency.py --dtype bfloat16 --enable-chunked-prefill False --load-format dummy`;BS ∈ {1,16,64},input_len ∈ {128,1024,2048},output_len ∈ {1,128,1024};模型为 Qwen2-7B / Phi-3-medium-128k / Meta-Llama-3.1-8B / Mistral-7B 的 w8a8 int8 量化版;2 次 warmup、5 次迭代取均值;硬件为 AMD GPU,具体型号未注明)

- 全模型全配置平均 2.8x speedup(完整 CSV 以附件 heuristic_speedups.csv 提供)。
- Qwen2-7B-Instruct-quantized.w8a8 示例:BS=1,in/out=128/128:1.4206→0.9828ms(1.45x);BS=1,1024/1024:11.4586→7.8414ms(1.46x);BS=64,2048/128:14.2707→4.7842ms(2.98x)——batch/序列越长收益越大。

## 验证与回归风险

验证依赖 benchmark_latency 延迟对比与全量 CSV;PR 未列单测或精度评测(launch 配置选择不影响数值结果,风险在于非最优 tile 导致的性能回退)。`use_heuristic=True` 为默认,改变 AMD 平台所有 compressed-tensors int8 GEMM 的 launch 配置;NVIDIA 平台不使用该 Triton 路径,不受影响。若某形状下启发式劣于 32x32x32,可用 `use_heuristic=False` 回退。

## 关联

启发式来源:CUTLASS `scaled_mm_c3x_sm90_int8_dispatch.cuh:116`(NVIDIA SM90 int8 scaled_mm dispatch)。pack 素材未列出其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
