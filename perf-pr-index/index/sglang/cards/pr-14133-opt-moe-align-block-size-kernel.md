# PR #14133: Opt moe align block size kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2025-12-02 |
| 分类 | MoE (tier: major) |
| Label | sgl-kernel, run-ci |
| Diff | 素材未提供 diffstat;diff 涉及 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/14133 |

## 一句话结论

优化 `moe_align_block_size` kernel:并行填充 + 小批量动态 padding,小批量提速 3-5.5%、分配内存最多省 96.8%。

## 动机

灵感来自 vLLM PR #29642。此前 `moe_align_block_size` 的 `sorted_token_ids` 初始化串行进行,且 `max_num_tokens_padded` 一律按 `topk_ids.numel() + num_experts * (block_size - 1)` 计算;当 token 数远小于 expert 数时(如 8 token / 256 expert),按旧公式要分配 32,520 个元素而实际只需 1,024 个,初始化开销与内存分配都被放大。这在小批量推理路径上是可直接观测的延迟。

## 优化点

两处机制改动:其一,把 `sorted_token_ids` 的填充从主计算路径剥离——大批量路径 grid 从 1 个 block 改为 2 个 block,`blockIdx.x == 1` 专门做向量化(VEC_SIZE)填充,与统计/排布逻辑并行;小批量路径则在同一 block 内划出 `fill_threads=256` 个线程先填充、再用三次 `__syncthreads()` 与其余线程同步。其二,当 `topk_ids.numel() < num_experts + 1` 时改用 `topk_ids.numel() * block_size` 计算缓冲大小,显著缩小分配。小批量 expert kernel 同时适配了新的线程划分(`tid = threadIdx.x - fill_threads`、`stride = blockDim.x - fill_threads`)。

## 关键改动

```text
sgl-kernel/csrc/moe/moe_align_kernel.cu: 主 kernel 拆出并行填充 block;小批量 kernel 加 fill_threads 线程组;host 传 max_num_tokens_padded
sgl-kernel/benchmark/bench_moe_align_block_size.py: benchmark 按 provider 区分动态 padding 与旧公式,对齐真实调用方式
sgl-kernel/tests/test_moe_align.py: 测试同步采用动态 max_num_tokens_padded 计算式
```

```cuda
if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      Vec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
      int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
    }
    return;
}
```

## 性能声明

小批量场景(tokens < experts):内存分配最多减少 **96.8%**(8 tokens、256 experts:32,520 → 1,024 元素),性能提升 **3-5.5%**。num_tokens=1 的示例(kernel µs):8 experts 7.93 → 7.53(5.0%),128 experts 9.02 → 8.63(4.3%),256 experts 9.19 → 8.68(5.5%)。中/大批量场景性能保持稳定、无回归。正文附 main 与 PR 两份完整 micro benchmark(num_tokens 1–8192,num_experts 8–256,topk 1–8,对比 SGL / SGL Fusion / Triton),PR 未注明测试硬件。另附 Kimi K2 Thinking TP8 服务下 GSM8K(2,000 题、8-shot)accuracy 0.939 作为正确性佐证。

## 验证与回归风险

依赖 `sgl-kernel/tests/test_moe_align.py` 的实现对比测试,且 benchmark 的差分校验同步改为动态 padding 公式;PR 注明需 bump sgl-kernel 版本并配合应用 #14134。回归风险:缓冲大小契约改变——调用方必须同步用新的 `numel * block_size` 公式分配 `sorted_token_ids`,否则与 kernel 期望不一致;小批量 kernel 线程划分重构(填充线程与计算线程共存于一个 block,依赖三次 `__syncthreads()`),以及大批量路径 grid 由 1 block 变 2 block,均需保证跨架构行为一致。

## 关联

灵感来源 vLLM PR #29642;需配套 #14134(sgl-kernel 版本相关)。无其他同区域 PR 编号记录。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
