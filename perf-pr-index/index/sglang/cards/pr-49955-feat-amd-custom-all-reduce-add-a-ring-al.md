# PR #49955: [Feat] [AMD] custom-all-reduce: Add a ring all-reduce and an unrolled 2shot_push on gfx1250

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenkaidu |
| 合并 | 2026-11-15 |
| 分类 | Parallelism / Communication (tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 5 个文件、其中 3 个新文件,另有 7 个文件因篇幅略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/49955 |

## 一句话结论

gfx1250 的 carv2 新增 ring_simple(每 CU 一个 channel 的 ring all-reduce)与 unrolled 2shot_push(更深 memory pipeline),并按 `HIP_FORCE_DEV_KERNARG` 两档重调尺寸表,使 carv2 覆盖 TP2-8 全部尺寸至 4 GB 而无 RCCL 回退;DeepSeek-V4-Flash TP8 serving 对 RCCL 提升 +3.1% 到 +7.8%。

## 动机

gfx1250 上 carv2 的 2shot_push 把几乎整个张量跨 fabric 送两次,per-vector kernel 在每次 scoped store 后等待;几 MB 以上输给 RCCL,大 all-reduce(prefill chunks、大 batch decode)回退 RCCL。本 PR 新增两条 gfx1250-only 路径让 carv2 承接 TP2-8 全部尺寸至 4 GB,并为 `HIP_FORCE_DEV_KERNARG=0`(默认)与 `=1` 两档 kernel 参数模式分别调尺寸表。`:0` 表的 TP8 行经 serving 验证;`:1` 表的 TP2/TP4/TP8 行在三个 8-GPU gfx1250 节点上验证。

## 优化点

- `ring_simple`:每 channel 256 线程、unroll 32,把 2 MB 切片经两槽 FIFO 推进下一 rank 内存,靠 64-bit ready/drained counters;FIFO 与 counters 在独立的 uncached symmetric buffer(#49382 的 SymmetricBuffer),每 channel 4 MB,channel 数等于 CU 数;ring 保持同加速器的分区相邻(否则 64 MB 起 1.5-1.8x 慢);每 channel 的张量份额按整 warp hunk 切分(修复偶数均分导致的 partial hunk——TP8 上非 12 MB 倍数的尺寸曾多付一整个 pass,40 MB 278 µs、48 MB 185 µs);`RingSimpleAllReduce` 与 `RingPlaneObj` 自成 FFI 对象,不进共享抽象,`AllReduceConfig.use_ring` 标记。
- unrolled `2shot_push`:保留 slots、Lamport markers、单字 push epoch(#49365)与 fp32 rank order;warp 先发整 hunk 的 load 再用,peer payload 批量 store;world size ≤8,阈值 TP8 1 MB、TP5-7 2 MB、TP2-4 8 MB(以下与 TP16 per-vector kernel 更快);由 `custom_all_reduce.cuh` 按尺寸选 kernel。
- `st_relaxed_run_16B`(ptx.cuh):8 条 system-scope `global_store_b128` + 一条 `s_wait_xcnt`(builtin store 令编译器在 gfx1250 上每 store 后等待,代价这些 kernel 4-19%,1-256 MB)。
- 尺寸表(TP2-8):1shot_push → 2shot_push → ring_simple 到 4 GB,中间无 RCCL 窗口;两套表(`=0`/`=1`)按 `HIP_FORCE_DEV_KERNARG` 选择。`:0` 表的 serving 例外:1shot_push 停在 64 KB(TP8 上 256 KB cutoff 曾损失 DeepSeek-V4-Flash C16 decode 吞吐的 18%);TP8 16 MB 留给 ring_simple(2shot_push 在该尺寸使每 rank 停顿约 2 ms:quant-after-all-reduce 复现中 `=0` 下首个 fp8 quant kernel 约 2 ms、`=1` 下约 51 µs)。

## 性能声明

(数字全部照抄 PR 正文,未独立核实。standalone 与 serving 表来自节点 B 最终 head,`=1` 表,bf16,CUDA graph,median of 3,max over ranks,µs/call;每格为 carv2 选择 / RCCL。)

| size | TP2 | TP4 | TP8 |
|---|---|---|---|
| 4 KB | 1shot 2.4 / 2.9 | 1shot 5.9 / 7.0 | 1shot 6.4 / 9.2 |
| 256 KB | 1shot 2.9 / 3.5 | 1shot 7.1 / 8.6 | 1shot 9.5 / 15.5 |
| 1 MB | 1shot 4.4 / 6.7 | 1shot 12.0 / 21.6 | 2shot 17.1 / 42.5 |
| 16 MB | 2shot 26.4 / 34.6 | 2shot 67.5 / 69.1 | 2shot 88.6 / 92.0 |
| 64 MB | ring 82.2 / 132.2 | ring 186.1 / 213.8 | ring 250.7 / 311.4 |
| 256 MB | ring 318.6 / 345.4 | ring 611.4 / 856.3 | ring 725.2 / 1011.9 |

"carv2 is faster than RCCL in every cell." DeepSeek-V4-Flash TP8 serving(mean of 2 alternating runs):8k/1k C16 788.9 → 816.6 tok/s(+3.5%);1k/1k C256 6,832 → 7,186(+5.2%);C64 3,334 → 3,594(+7.8%);C16 1,196 → 1,233(+3.1%);TTFT 降 4-6%、TPOT 降 3-7%。`=1` 表相对 `=0` 表:TP4 +2.3-4.1%、TP8 +1.9-3.8%(1shot_push 进 decode)。GSM8K:carv2 0.944 / 0.949 vs RCCL 0.942 / 0.945(各 2 runs)。

## 验证与回归风险

`test_custom_all_reduce.py` 对 RCCL bit-exact(atol=0, rtol=0),eager 与 CUDA graph,fp16/bf16/fp32:TP2/TP4/TP6/TP8 各 443 passed(最终 head 在节点 B 复验,ring 与 small-FIFO ring 用例在内);capability/fabric 单测 84 passed(merge main 前为 90,#49382 重构了 capability 测试);TP16 hier 路径(两主机)未在此栈重跑;`=0` 表 TP8 行在节点 C、`=1` 表 TP2/TP4/TP8 在节点 A 验证,TP8 又在节点 B 复验。CI 状态:Base 通过,Extra 与 AMD ROCm 10 失败。回归风险:(1) 第六个 commit 后未重调 cutoff:TP8 32 MB ring(157 µs)已胜过表选的 2shot_push(173 µs),TP4 32 MB RCCL(112 µs)胜过两者(129/117 µs)——DeepSeek-V4-Flash 未达该尺寸;(2) 两套表并存的心智负担与 `SGLANG_FORCE_*` 强制路径需重验;(3) uncached symmetric memory 依赖 #49382;(4) cache hints(NT load/store)被测无增益后移除,但正文注明部分早前测量带着 hints。

## 关联

#49377(被本 PR 取代)、#49370(其 base)、#49382(shared symmetric workspace)、#49365(push epoch);后续:#50660(hier_ring_simple)与 #50656(clamp 移除 + 重调,均在本批次)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
