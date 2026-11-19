# PR #50660: [Perf] [AMD] custom-all-reduce: Add hier_ring_simple, a ring all-reduce over partition pairs on gfx1250

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenkaidu |
| 合并 | 2026-11-16 |
| 分类 | Parallelism / Communication (tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 4 个文件,另有 4 个文件因篇幅略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/50660 |

## 一句话结论

为 gfx1250 DPX 分区对新增分层 ring all-reduce `hier_ring_simple`:每个 chunk 按两分区对半切,分区内的环与分区间独立链路在同一 kernel 内传递,每加速器外发量从 2*(2m−1)/2m 降到 2*(m−1)/m;64-256 MB 比 ring_simple 快 10-17%、比 RCCL 快 28-41%。

## 动机

gfx1250 加速器拆成两个分区时,分区共享到其他加速器的链路,而两分区之间是一条独立链路。8-GPU 节点(bf16、CUDA graph、256 MB)上,两个跨加速器的 TP2 ring 并行跑(GPU 0/2 与 1/3)要 634-777 µs(单 ring 448 µs),而加速器内两个 ring(GPU 0/1 与 2/3)都保持 319 µs。`ring_simple`(#49955)已把同加速器的分区在环上放相邻,但每个加速器仍外发 2 * (2m − 1) / 2m 的张量(m 个加速器,TP8 时 1.75x 张量);本 PR 让每个 chunk 跨两分区对半切,每加速器只发 2 * (m − 1) / m(TP8 时 1.5x),pair 的份额经分区间链路在同一 kernel 内传递。

## 优化点

- kernel(`csrc/distributed/ring_simple.cuh` 的 `all_reduce_hier_ring_kernel`):一次 launch、每 CU 一个 channel;每个 chunk 的半 h 跑 partition h 的环,每 loop 2m − 1 步;每 channel 3 个 FIFO(cross / pair-in / pair-out),像 ring_simple 一样用 64-bit ready/drained counters;step s 期间块的 8 个 warp 中 3 个搬 peer 的 step s+1 pair-in chunk 与 step s−1 的 pair-out chunk,其余 5 个跑 cross step(所有 warp 顺序执行时,pair copies 占 TP8 256 MB 时间的五分之一);每 thread 每源 16 packs(reducing 步要读 3 个源,ring_simple 的 32 个 spill 寄存放不下;16 比 32 或 8 在 TP4/TP8 64 MB 起快 5-15%)。
- `HierRingPlane`(FFI 对象)与 `HierRingSimpleAllReduce`(`ring_simple_all_reduce.py`):plan、独立的 uncached symmetric buffer(每 channel 3 FIFO x 2 x 512 KB slices)与 all-reduce。
- Dispatch:`Heuristic.hier_ring_simple` 紧跟 hier_2shot_push 之后尝试;plan 只对由分区对构成(至少两个加速器)的组保留,此时 2shot_push 只留其以下尺寸、ring_simple 不再保留被覆盖的尺寸,flat ring plane 不再分配;其他组拿到原表。
- 表:目前只有 `HIP_FORCE_DEV_KERNARG=1` 行启用:TP4 从 8 MB + 16、TP8 从 16 MB + 16 起开始胜过 2shot_push。TP2 是单加速器;TP16 未测。

## 关键改动

```text
python/sglang/kernels/jit/csrc/distributed/ring_simple.cuh:
+inline constexpr uint32_t kHierRingUnroll = 16;
+inline constexpr uint32_t kHierCrossWarps = 5;
+__global__ ... void all_reduce_hier_ring_kernel(...)  // cross + pair-in + pair-out 三 FIFO,2m-1 步/loop
+SGLANG_REGISTER_FFI_REFERENCE_CLASS(HierRingPlaneRef, HierRingPlaneObj);
python/sglang/kernels/ops/communication/ring_simple.py:
+def custom_hier_ring_simple(ring: HierRingPlane, input: torch.Tensor) -> torch.Tensor:
python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py:
+    hier_ring_from = {4: 8 * MB + 16, 8: 16 * MB + 16} if dev_kernarg else {}
python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py:
+    if heuristic.hier_ring_simple.contains(nbytes):
+        return AllReduceChoice(None, use_hier=True, use_ring=True)
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实。8-GPU gfx1250 节点,每 GPU 两个分区,bf16,每 CUDA graph 8 个背靠背 all-reduce,10 replays,median of 3,max over ranks,µs;每格为 ring_simple / hier_ring_simple / RCCL。)

| size | TP4 | TP8 |
|---|---|---|
| 16 MB | 72.9 / 63.9 / 68.5 | 145.8 / 91.9 / 91.3 |
| 32 MB | 117.9 / 92.3 / 112.0 | 158.3 / 140.1 / 180.2 |
| 64 MB | 187.2 / 154.6 / 214.0 | 253.2 / 211.2 / 310.6 |
| 128 MB | 335.2 / 281.2 / 417.3 | 413.1 / 358.4 / 611.5 |
| 256 MB | 613.5 / 544.8 / 856.5 | 729.1 / 659.0 / 1011.5 |

"From 64 MB to 256 MB, `hier_ring_simple` is 10-17% faster than `ring_simple` at TP4 and TP8, and 28-41% faster than RCCL." TP6(3 加速器):16 MB 168.6 / 99.8 µs,64 MB 253.9 / 182.8,256 MB 684.3 / 615.6(ring_simple / hier_ring_simple)。对 2shot_push(=1 表在这些 cutoff 以下的选择):TP4 从 12 MB 起赢(50.3 vs 55.1 µs),TP8 从 20 MB 起(97.7 vs 111.9 µs);2shot_push 在 TP4 8 MB、TP8 16 MB 仍领先。rebase 到 main 后第二节点与第一节点的差距在 1 µs 内(TP8:64/256 MB 为 211 / 659 µs;TP4:154 / 545 µs)。DeepSeek-V4-Flash serving(`HIP_FORCE_DEV_KERNARG=1`、`SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1`、单 run)在 run-to-run spread 内不变:TP4 8k/1k C16 768.1 → 763.9、1k/1k C256 6,495 → 6,537 tok/s;其 all-reduce 至多 16 MB,TP8 serving 达不到其范围。

## 验证与回归风险

`test_custom_all_reduce.py` 对 RCCL bit-exact(atol=0, rtol=0),eager 与 CUDA graph,fp16/bf16/fp32,在 main 加本 PR 上:575 passed at TP4/TP6/TP8(`HIP_FORCE_DEV_KERNARG=1`)、TP8(`=0`)、TP8 且 `HIP_VISIBLE_DEVICES=0,2,4,6,1,3,5,7`(一个加速器的分区分开在组内);TP2 443 passed(hier ring 用例跳过,单加速器)。capability 与 fabric 单测:97 passed。新增 hier ring 测试(512 KB 与 2 KB slices,eager 与 CUDA graph)。CI 状态:Base 通过,Extra 与 AMD ROCm 10 失败。回归风险:(1) 仅分区对组、仅 =1 表启用,其他组合回退原表;(2) 每 channel 3 FIFO x 2 x 512 KB 的新 symmetric buffer 内存;(3) `HierRingSemaphore` 计数器跨 launch 持久,首次使用前必须零填充;(4) TP16 未测、TP2 单加速器不支持(world_size % 2 == 0 且 >= 4 的校验)。

## 关联

#49955(ring_simple 基础,本批次)、#50656(在其上 retune 并扩展到 TP16,本批次)、#49382(shared symmetric workspace)、#49955 的两个 commit 结构。hier_2shot_push 出自 #49205(DpxHierAllReduce)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
