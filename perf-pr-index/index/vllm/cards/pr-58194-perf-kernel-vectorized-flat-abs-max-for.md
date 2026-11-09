# PR #58194: [Perf][Kernel] Vectorized flat abs-max for dynamic per-tensor FP8 quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Monishver11 |
| 合并 | 2026-09-25 |
| 分类 | MoE \| tier: major |
| Label | performance, quantization, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/58194 |

## 一句话结论

per-tensor 动态 FP8 量化的 abs-max 改向量化 flat + warp shuffle 归约，geomean 1.39x、端到端 +2.7% 吞吐。

## 动机

`dynamic_scaled_fp8_quant` 的 abs-max pass（csrc/libtorch_stable/quantization/w8a8/fp8/common.cu）是 `--quantization fp8` 服务模型里每个 MoE 层的 activation quant，从未做过性能调优。作者读了 #44173（把同类 kernel 从 block-per-row / shared-memory 树归约 / 标量 load 改写为 warp shuffle + 向量 load）后用同样方法重写。原 `segmented_max_reduction_strided` 每 token 一个 block、每元素一次标量 load、256-entry shared-memory 树每层一次 __syncthreads、每 token 一次 atomic：decode batch 32 tokens 在 132-SM GPU 上只有 32 blocks，65536-token prefill 则对同一地址发 65536 次 atomic。

## 优化点

contiguous 输入下 abs-max 改为对 16-element vectors 的 flat grid-stride kernel（bf16 32 B / fp32 64 B 向量 load），归约用 warp 内 __shfl_xor_sync butterfly 加跨 warp 一次 shared-memory exchange（单次 barrier），grid 按向量数并上限 8 blocks/SM，每 block 一次 atomic 而非每 token。strided / unaligned 输入保留 one-block-per-token kernel，但改用 vectorize_read_with_alignment 的向量 load 与同样的 shuffle 归约替代树归约；后面的 quantize pass 不变。正文说明输出与 main bit-for-bit 相同（fmaxf 顺序无关、除以正常数单调，故重新分组 max 不改任何 scale 或字节），无 API 变化。

## 关键改动

```text
csrc/libtorch_stable/quantization/w8a8/fp8/common.cu: 新增 warp_reduce_max / block_reduce_max / absmax_reduction_flat，重写 abs-max
tests/kernels/quantization/test_fp8_quant.py: 新增强制走 fallback kernel 的 strided-input 测试
```

```cpp
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, VLLM_SHFL_XOR_SYNC(v, offset));
  }
  return v;
}

// Per-tensor abs-max of a contiguous tensor viewed as a flat array of
// 16-element vectors: grid-stride loop, one atomic per block.
__global__ void __launch_bounds__(kFp8QuantBlockSize)
    absmax_reduction_flat(float* __restrict__ scale,
                          const scalar_t* __restrict__ input,
                          int64_t num_vecs) {
```

## 性能声明

Per-tensor bf16 main→PR（us）：16/4096 8.21→5.61（1.46x）、16/8192 11.36→6.01（1.89x）、512/4096 9.58→7.39（1.30x）、4096/4096 48.95→30.51（1.60x）、16384/4096 187.22→115.96（1.61x）、65536/4096 715.88→436.15（1.64x）、65536/8192 1378.42→866.10（1.59x）；Geomean 1.39x（bf16/fp32、M∈{16,512,4096,16384,65536}、hidden∈{2048,4096,7168,8192}），范围 1.11x-1.89x，大 M 时约 3.0 TB/s 实际流量。端到端 Qwen3-30B-A3B `--quantization fp8`、`vllm bench throughput`（random、1024 in/128 out、512 prompts、3 次交错）：main 43793/43706/43835（mean 43778 ±66）→ PR 45154/44810/44965（mean 44977 ±172），**+2.7% throughput**，run-to-run spread <0.4%；`vllm bench latency` batch 32：1.876 s → 1.843 s avg（约 1.8% lower）。测试条件：H100 80 GB HBM3、driver 580、CUDA 13.0.3、torch 2.13.0+cu130，triton.testing.do_bench_cudagraph（取含 memset 与两个 kernel 的完整 op 中位）；模型 Qwen/Qwen3-30B-A3B，精度 online per-tensor dynamic FP8，workload vllm bench throughput / latency。

## 验证与回归风险

Bitwise：bf16/fp32、M∈{1,7,64,512,4096,16384}、hidden∈{17,1024,1025,2048,4096,5137,7168,8192,16384,32768}、三种 row layout（contiguous / misaligned by 8 / aligned padded by 16），1080 cases identical、0 mismatched（bytes 与 scales）；单测 249 passed；pre-commit 与 mypy-3.12 通过。回归风险：输出 bit-for-bit 相同、无 API 变化；strided / unaligned 走 fallback kernel；后续 quantize pass 不变；仅影响 `--quantization fp8` 的 per-tensor dynamic quant / MoE expert input 路径；正文计划后续 PR 用此处新增的归约 helper 处理同文件的 per-token kernel。

## 关联

#44173（同类 kernel 的 warp shuffle + 向量 load 改写思路）、#41115（调 per-token kernel block sizes，不碰 per-tensor path）、#55330（per-token-group kernel）、#45428（fused RMSNorm quant kernel）；正文另说明合并后计划提 per-token kernel 的 follow-up PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
