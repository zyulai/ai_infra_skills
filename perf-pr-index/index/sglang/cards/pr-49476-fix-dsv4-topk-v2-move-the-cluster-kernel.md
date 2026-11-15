# PR #49476: [Fix] dsv4/topk_v2: Move the cluster kernels' top-k buffer to dynamic shared memory

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @akhilg-nv |
| 合并 | 2026-11-11 |
| 分类 | Attention / Kernel(tier: major) |
| Label | jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/49476 |

## 一句话结论

把 dsv4 `topk_v2.cuh` 两个 cluster kernel 的 8 KiB top-k indices buffer 从 static 移到 dynamic shared memory,修复 ptxas 在 CUDA 12.9(`sm_90a`/`sm_100a`)与 family target(`sm_100f`)下的 "uses too much shared data" 构建失败;B200 微基准 16 个形状中 8 个快 0.3–3.8%,无形状变慢。

## 动机

`topk_v2.cuh` 是 `sglang.kernels.ops.attention.dsv4.topk` 背后的 JIT top-k(DeepSeek-V4 sparse-attention indexer 的 top-k,也被 GLM k-pool indexer 的 `topk_transform_kpool_v2` 使用)。部分 toolkit/target 组合下 ptxas 拒绝整个模块,首次调用(例如 `plan_topk_v2()`)即 raise。原因是两个 cluster kernel(`topk_small_batch_cluster_kernel`、`topk_candidates_cluster_kernel`)各声明两个 static shared buffer:`impl::TopKCluster<C>::Smem`(cluster size C=2–16 时 41.6–47.2 KiB)与 `int32_t s_topk_indices[kMaxTopK]`(8 KiB),合计 49.6–55.2 KiB/block,超过 48 KiB static 上限(CUDA Programming Guide:超过 48 KB 必须用 dynamic shared memory 并显式 opt-in)。ptxas 实测:CUDA 12.9 对 `sm_90a`/`sm_100a`/`sm_100f` 均拒绝;CUDA 13.0/13.4/13.5 仅 `sm_100f` 拒绝(接受 `sm_90a`/`sm_100a`)。#49385 报告了 CC 10.7 上的该失败,#49420 通过构建 `sm_107a` 绕过;本 PR 修 kernel 本身,覆盖 target selection 无济于处的组合(CUDA 12.9 的 `sm_90a`/`sm_100a`,以及 `sm_100f` 这类 family target)。

## 优化点

- 两个 kernel 中 `s_topk_indices` 改为 `extern __shared__ __align__(16) int32_t s_topk_dyn[]` 的视图;每次 launch 传 `kMaxTopK * sizeof(int32_t)`(8 KiB)dynamic shared memory,并经 `runtime::init_per_device` 每 device 一次设置 `cudaFuncAttributeMaxDynamicSharedMemorySize`(与既有 C16 cluster-size attribute 同法)。static 部分只剩 `TopKCluster<C>::Smem`:`ptxas -v` 报 0xa650 / 0xa980 / 0xafe0 / 0xbca0 字节(C = 2/4/8/16),main 为 0xc650 / 0xc980 / 0xcfe0 / 0xdca0。对超限实例,static + dynamic 每 block 总量与 main 相同。
- `device::enable_smem_spilling()` 从这两个 kernel 移除(PTX ISA 不允许在使用 dynamic shared memory 的 kernel 里用该 pragma;helper 仅在 CUDA 13+ 发出该 pragma)。register counts 前后一致;CUDA 13.4 `sm_90a` 一个 k-pool C8 instance 的 spill-store 86→74 bytes;CUDA 13.0 `sm_100a` 三个 C8 small-batch instance spill 4 bytes stored / 32 loaded(main 为 8/2 或无 spill)。
- small-batch kernel 的两个 index-only instance(无 page table)从不使用该 buffer、从未超限,现在也带 8 KiB dynamic smem,per-block 总量 +8 KiB(static 48,288 / 45,024 B 不变,`ptxas -v` 统计不变)。
- 同步不变:k-pool 模式下 cluster 各 rank 仍经 `cluster.map_shared_rank(s_topk_indices, 0)` 写 rank 0 的 buffer、`cluster.sync()` 后 rank 0 读取(dynamic buffer 下每 block 布局相同,map_shared_rank 仍指向 rank 0)。
- 剩余余量:C16 static 现为 0xbca0 = 48,288 字节,距 0xc000 上限 864 字节;若 `TopKCluster<16>::Smem` 增长超过 864 字节,CUDA 12.9 与 family target 下会再次构建失败(CUDA 13 `sm_90a`/`sm_100a` 构建不会显现)。正文列出未做的后续:`runtime::set_max_dynamic_smem_per_device<kernel>()` 可替代三段手写 `cudaFuncSetAttribute`。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh: 两个 cluster kernel 的 s_topk_indices 移至 dynamic smem;移除 enable_smem_spilling();四处 launch 传 8 KiB dynamic smem 并设置 MaxDynamicSharedMemorySize
```

正文注明 **+36 / −9 runtime code(1 file),无测试改动;无新 kernel**。

```cpp
// 两个 cluster kernel 内(以 small-batch 为例)
-  device::enable_smem_spilling();
-  __shared__ int32_t s_topk_indices[kMaxTopK];
+  // The kMaxTopK buffer is dynamic smem: shared memory over 48 KB per block must be
+  // dynamic (CUDA Programming Guide); enable_smem_spilling() is omitted, because the
+  // PTX ISA forbids it in kernels that use dynamic smem.
+  extern __shared__ __align__(16) int32_t s_topk_dyn[];
+  int32_t* const s_topk_indices = s_topk_dyn;

// launch 处(每个改动 kernel)
runtime::init_per_device(device, [kernel_ptr] {
  CHECK_CUDA(::cudaFuncSetAttribute(
      kernel_ptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kMaxTopK * sizeof(int32_t))));
});
return LaunchKernel({batch_size, kClusterSize}, kBlockSize, device,
                    kMaxTopK * sizeof(int32_t))
    .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
    .launch(kernel, params);
```

## 性能声明

- B200(CUDA 13.0,默认 `sm_100a` 构建)微基准:small-batch kernel 各输出模式 + candidates kernel,16 shapes;`marker.do_bench` CUDA-graph 模式、`replay_iters=200`、每次 replay 前 flush L2、每进程测一棵树、8 进程按 PR,main, main, PR, main, PR, PR, main 排序,表值为每进程 median 的 mean(µs,括号为 4 进程 min–max)。8/12 的 page-table、dual-output、k-pool 形状快 0.3%–3.8% 且区间不重叠(如 page-table C8 30x131072 k=2048:12.303→11.831 µs,−3.84%;dual-output C16 1x65540:5.279→5.243 µs,−0.69%;k-pool offsets C8 8x65536 pool k=2048:8.873→8.760 µs,−1.27%);index-only 与 candidates 形状不变;最大单次收益 0.47 µs/call;无形状慢于 main 超出进程间散布。正文注明"为何变快未查明":第二次运行验证被移除的 pragma 不是原因(main 仅去掉两处 `enable_smem_spilling()` 时各形状与 main 差 ≤0.11% 量级),差异随 dynamic buffer 本身而来。同机第二次运行(12 进程)把每个差异复现到 0.25 个百分点内(噪声的 8x65536 k-pool 形状除外,−0.07%)。
- 端到端:未跑;正文估计无 slowdown(被改 kernel 无一形状慢于 main,其余不变)。
- 精度:`test_topk_v2.py` B200(CUDA 13.0.88,torch 2.14.1+cu130):402 passed、14 skipped(ROCm-only)、0 failed,与 main 相同;强制 `sm_100f` 时 main 构建失败(8 个 ptxas "uses too much shared data" 错误)而本 PR 通过同样 402 测试。所选 index 集合与 main 在 52/52 用例相等(按 multiset 含 −1 padding 比较);48 个 small-batch 用例的每一行同时匹配 `torch.topk` reference。输出槽位顺序不 bitwise 可复现:main 自身两次 run 在 52/52 用例 bitwise 不同而集合一致。

## 验证与回归风险

- Mutant 检查(每用例独立进程):三处 launch 改传 4 字节 dynamic smem 时,small-batch 与 candidates kernel 的测试 illegal memory access、6 个 k-pool 用例运行时 CUDA 错误;k-pool ranks 改写 rank 1 的 buffer 时,6/6 k-pool 用例未通过 selected-set 检查——证明测试确实触达被改代码。
- Compile only(无 GPU):本改动下模块在 CUDA 12.9.86(`sm_90a`/`sm_100a`/`sm_100f`)、13.0.88(`sm_100f`)、13.4.92(`sm_90a`/`sm_100a`/`sm_100f`)均可构建;CUDA 13.5.37 仅在 main 上检查过(`sm_100f` 失败);其他 toolkit 未检查。
- 风险:剩余 864 字节余量(`TopKCluster<16>::Smem` 再涨即复发);k-pool 检查用的是 standalone pytest 文件,未加入本 PR(`test_topk_v2.py` 无 k-pool case);强制 `sm_100f` 的 pytest plugin 不属于本 PR;三个 `test_topk_v2_modes` 子进程用例仍按 `sm_100a` 构建。正文列出未采纳的后续:`runtime::set_max_dynamic_smem_per_device<kernel>()` 统一三段 attribute 代码(以"测试过的代码即交付的代码"为由未做)。

## 关联

#49385、#49420(正文 Related:同一构建失败的报告与 #49420 的 target-selection 绕过);#47960(16 KiB `TieValue members[kMaxNumTie]` 数组来源)、#48501(`topk_candidates_cluster_kernel` 来源)为正文提及的历史来源。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
