# PR #7884: [kernel] opt moe align block kernel by block/warp scan algorithm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-07-17 |
| 分类 | MoE |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);单文件 sgl-kernel/csrc/moe/moe_align_kernel.cu |
| 链接 | https://github.com/sgl-project/sglang/pull/7884 |

## 一句话结论

把 fused MoE 路径 `moe_align_block_size_kernel` 的 Blelloch 扫描换成 block/warp 两级 shfl 扫描,自述约 10% kernel 提速。

## 动机

fused MoE 每层前向都要调 `moe_align_block_size_kernel` 把 topk_ids 按 expert 计数、对齐 block 并生成前缀和/排序索引,是 MoE 路径上的固定小 kernel。旧实现用 shared memory 上的 Blelloch scan(up-sweep + down-sweep),每级都要 `__syncthreads()`,对 num_experts 对应的 scan_size 反复做全块同步;该 kernel 本身耗时极短(约 20 µs 量级),同步次数是主要开销来源。PR 正文未附 profile,直接以替换扫描算法为目标,自述约 10% 提速。适用范围:num_experts ≥ 128 走本 kernel;num_experts < 128 走 `moe_align_block_size_small_batch_expert_kernel`(不受本 PR 影响)。

## 优化点

以两级 warp scan 替换 Blelloch scan:新增 `warp_exclusive_scan`(基于 `__shfl_up_sync` 的蝶形 exclusive scan,HIP/CUDA 经 `SHFL_UP` 宏统一),scan 按行组织为"各 warp 内扫描 → warp 0 对 warp_sums 再扫描 → 每线程加自己 warp 的偏移"两级结构,`__syncthreads()` 次数从每级一次(约 2×log(scan_size) 次)降到固定数次;`total_tokens_post_pad` 在 warp0 汇总阶段由 tid==0 一次写出,不再依赖 scan_buf 末元素;shared memory 布局追加 `warp_sums`(≤32 int32)。同时调整 host 侧 `shared_mem_size` 并删除不再需要的 `padded_num_experts`/`experts_per_warp`。count_and_sort_expert_tokens_kernel 不变;CUDA 图形路径兼容(scan 全部在同一 kernel 内完成)。

## 关键改动

```text
sgl-kernel/csrc/moe/moe_align_kernel.cu: 新增 warp_exclusive_scan;moe_align_block_size_kernel 由 Blelloch scan 改为两级 warp scan;shared memory 增加 warp_sums
```

关键机制(warp 级蝶形 exclusive scan):

```cpp
__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = SHFL_UP(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}
```

## 性能声明

PR 正文 benchmark(`sgl-kernel/benchmark/bench_moe_align_block_size.py`,PR 未注明 GPU/硬件;单位 µs,SGL 列 main vs 本 PR):

- 自述 "approximately 10% speedup"。
- 代表性档位:1 token / 128 experts:20.448 → 19.296;1 token / 256 experts:20.576 → 19.232;64 tokens / 128 experts / topk=1:19.360 → 19.360(持平);8192 tokens / 8 experts / topk=8:SGL 54.848 / SGL Fusion 57.056,Triton 同档 1443.904。小 batch(<128 experts 走 small_batch_expert_kernel 的档位)数值基本不变。
- 正确性交叉验证:SGL 与 Triton、SGL 与 vLLM 实现结果一致(bench 脚本内置 ✅ 断言)。

## 验证与回归风险

验证依赖 bench 脚本的 SGL/Triton/vLLM 三方一致性断言与分档耗时表;PR 未注明新增独立单元测试。风险点:两级 warp scan 依赖 `scan_size <= 1024`(num_experts ≤ 1024 的 next_pow2)且线程数 1024 的既有假设;`warp_sums` 复用同一 shared memory 区域,两次扫描间 `__syncthreads()` 的放置决定正确性;SHFL_UP 宏为 HIP/CUDA 双路径,ROCm 行为依赖宏展开正确;仅影响 num_experts ≥ 128 的 kernel,小 expert 数路径无变化。

## 关联

同区域(moe align / MoE kernel):#14133(opt moe align block size kernel,后续进一步优化)、#43363、#44268、#48440(AMD small batch sort)。同类扫描/排序 kernel 优化:#15347(topk routing)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
