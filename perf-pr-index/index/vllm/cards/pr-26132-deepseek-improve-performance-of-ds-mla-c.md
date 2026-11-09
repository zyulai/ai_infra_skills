# PR #26132: [DeepSeek] Improve performance of DS MLA cache kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2025-10-03 |
| 分类 | Attention / Kernel |
| Label | ready, deepseek |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/26132 |

## 一句话结论

重写 DeepSeek MLA fp8 KV cache 写入 kernel:每线程管 8 NoPE/2 RoPE 元素并向量化,block 576→96 线程,H100 大 batch 吞吐 +58%,最好约 60%。

## 动机

`concat_and_cache_ds_mla_kernel`(fp8_ds_mla KV cache 写入路径)原实现每个线程只处理 1 个 NoPE 元素,需要 18 个 warp(576 线程)per token block,scale 归约还要经 shared memory 两阶段完成并插入两次 `__syncthreads`;对这种带宽型 memcpy+量化 kernel,线程数与同步是主要浪费。

## 优化点

NoPE 部分改为每线程处理 8 个元素:一次 int4 向量化加载(8×16-bit 值),半个 warp(16 线程)负责一个 128 元素 tile;tile 的 max-abs 归约从"全 warp 归约 + shared memory 合并"改为半 warp 的 `VLLM_SHFL_XOR_SYNC_WIDTH` shuffle 归约,去掉 shared memory 数组与两次 `__syncthreads`;tile scale 由每半 warp 的 lane 0/16 直接写进 kv_cache;量化后的 8 字节按 64-bit 对齐一次性存储。RoPE 部分每线程处理 2 个元素(单次 32-bit load/store)。每 token block 线程数 576 → 96(3 warps)。量化语义不变:tile 仍是 128 元素,scale 仍为 tile max/448。

## 关键改动

```text
csrc/cache_kernels.cu: concat_and_cache_ds_mla_kernel 重写;launch block 尺寸 576 → 96
```

```cpp
// Each thread handles 8 elements of NoPE
const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
// Vectorized load of eight 16-bit values, performed as an int4 load
const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
...
// Warp-level reduction to find the max absolute value in each half-warp
for (int offset = 8; offset > 0; offset /= 2) {
  max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
}
float tile_scale = max_abs / 448.f;
...
// Store as aligned 64-bit writes
*reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
    *reinterpret_cast<const uint64_t*>(result);
```

## 性能声明

- 条件:NVIDIA H100 80GB HBM3,PyTorch 2.8.0+cu128,CUDA 12.8,fp16,每配置 1000 次迭代,kv_lora_rank=512 / qk_rope_head_dim=64 / block_size=16。
- 2048 tokens:mean 0.020 → 0.012 ms,吞吐 104,694,434 → 165,652,093 tokens/s(112.32 → 177.73 GB/s,约 +58%);1024 tokens:0.015 → 0.012 ms(71.40 → 94.51 GB/s);512 tokens:0.013 → 0.011 ms;42/128/256 tokens:0.012/0.012/0.012 → 0.011 ms。
- PR 正文声明 best-case 约 60% speedup;未提供 E2E serving 数字。

## 验证与回归风险

`pytest tests/kernels/attention/test_cache.py::test_concat_and_cache_ds_mla` 通过。风险:线程划分与 DeepSeek MLA 固定形状(kv_lora_rank=512、rope 64、fp8 e4m3 tile scale)硬耦合,仅 fp8_ds_mla 分支变化,非 fp8 的 concat/cache 路径不受影响;scale 归约语义从两阶段改为半 warp 直接写,依赖 tile 内 16 线程×8 元素恰为 128 的布局;PR checklist 未勾选,无 E2E/accuracy eval,性能数据为 kernel microbenchmark。

## 关联

素材未提及具体关联 PR。索引同区域另有 pr-13747(FlashMLA integration)、pr-21078(Flashinfer MLA trtllm-gen decode)、pr-20769(SM100 CUTLASS MLA decode)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
