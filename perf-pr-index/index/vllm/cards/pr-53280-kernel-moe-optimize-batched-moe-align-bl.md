# PR #53280: [Kernel][MoE] Optimize batched_moe_align_block_size with cooperative writes

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-09-11 |
| 分类 | MoE |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/53280 |

## 一句话结论

batched MoE 对齐 kernel 的 token/block-ID 写入改为按 warp 组协作并行,选中协作路径的负载在 H200 上提速 1.10x~8.00x。

## 动机

`batched_moe_align_block_size` kernel(生产调用方为 `BatchedMarlinExperts`,用于特定 expert-parallel 配置的 batched activation 格式)原实现是"每 batch 一个线程"的串行写:每个线程单独循环写出该 batch 的全部 `sorted_ids` 与 `block_ids`,batch 容量大时单线程写延迟主导整个 kernel。此前 #35435 与 #40370 提过类似协作写方案但均过期关闭。

## 优化点

kernel 模板化出 `cooperative_writes` 两个特化:(1) 协作路径——把 `cumsum/valid_tokens/batch_num_blocks` 先写入 shared memory,再将 1024 线程按 2 的幂分组(每组尽量一个 warp,`num_groups` 上限为 blockDim/WARP_SIZE),组间跨步认领 batch,组内按 `threads_per_batch` 跨步并行写 token 与 block-ID;(2) 串行路径——保留原逐 batch 单线程写。launch 侧用 H200 调优的 guard:`max_tokens_per_batch >= max(256, 8*B)` 才走协作路径,小容量/多 batch 场景保留原特化(二进制检查确认原路径仍为 48 寄存器、5280 B shared memory,未被协作分支的额外 shared state 拖累)。同时把 `sorted_ids_size` 等输出索引统一为显式 `size_t`,保留大容量正确性。

## 关键改动

```text
csrc/libtorch_stable/moe/moe_align_sum_kernels.cu: kernel 模板化 cooperative_writes + shared 状态 + 组间跨步写;launch guard
tests/kernels/moe/test_moe_align_block_size.py: 参数化扩展(block_size 48、token_distribution random/alternating_empty/skewed)
benchmarks/kernels/benchmark_moe_align_block_size.py: 新增 --batched 基准矩阵(4x4x4 分布 x 3 种 token 分布)
```

```cuda
for (int32_t batch_id = group_id; batch_id < num_batches;
     batch_id += num_groups) {
  size_t const cumsum = batch_cumsum[batch_id];
  for (size_t i = group_offset; i < valid_tokens[batch_id];
       i += threads_per_batch) {
    sorted_ids[cumsum + i] = static_cast<int32_t>(batch_offset + i);
  }
  ...
}
```

## 性能声明

NVIDIA H200 (SM90),guarded raw-kernel 基准矩阵(`benchmark_moe_align_block_size.py --batched`,num_batches/max_tokens_per_batch/block_size x uniform/skewed/mostly_empty 分布):选择协作路径的负载测得约 1.10x~8.00x speedup(单一区间声明,未逐项列数);guard 以下的负载走原特化,寄存器与 shared-memory profile 与原实现一致。作者说明模型评估未跑,因对齐算子输出 bit 级不变、不改模型数值。

## 验证与回归风险

`pytest tests/kernels/moe/test_moe_align_block_size.py -q`:525 passed;测试矩阵扩展覆盖 block_size=48 与 random/alternating_empty/skewed 三种 token 分布,`expert_num_tokens` 上界修正为 `max_tokens_per_batch + 1`(含满 batch)。新增 batched 基准矩阵便于后续回归对比。风险面:双特化由 guard 选择,`B <= num_threads(1024)` 仍是硬约束;协作分支新增 3 个 `__shared__` 数组(仅该特化分配),若未来调低 guard 阈值需重估 shared memory 占用;生产影响集中在 BatchedMarlinExperts 的 EP 路径。

## 关联

同区域(MoE 对齐/调度 kernel):索引内 PR #46340(batched MoE TD operand loads)、PR #28373(AMD Triton MoE tuning)、PR #28369(EPLB balance packing);作者提及的前序(未合并)方案 PR #35435、#40370。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
