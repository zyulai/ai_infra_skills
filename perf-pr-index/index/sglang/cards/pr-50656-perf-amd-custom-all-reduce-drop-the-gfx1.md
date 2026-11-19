# PR #50656: [Perf] [AMD] custom-all-reduce: Drop the gfx1250 2shot_push clamp and retune carv2

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw |
| 合并 | 2026-11-17 |
| 分类 | Parallelism / Communication (tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 6 个文件,无新文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/50656 |

## 一句话结论

移除 gfx1250 2shot_push 的 256-thread clamp(实测对偏斜无益、回背靠背昂贵),以新 kernel 重调 TP2-TP16 的 carv2 尺寸表、给 DPX pair 阶段配 unrolled kernel 并把 hier_ring_simple 扩到 TP16:TP4 4MB 延迟 80.5 → 25.6 µs(−68%),TP16 64MB 408.5 → 291.1 µs(−29%)。

## 动机

本 PR 是 #49205(hier_2shot_push)、#49955(ring_simple 与 unrolled 2shot_push)、#50660(hier_ring_simple)在 main 上的 follow-up。gfx1250 上每个 ≤4 MB 且未走 unrolled kernel 的 2shot_push 被 clamp 到 256 threads/block;注释称有助于 rank 到达偏斜,但在当前 kernel 上不成立:单 rank 迟到约 20 µs 时默认 block size 在所有尺寸下一样快或更快(host 与 device kernel args 均如此);回背靠背时 clamp 昂贵(forced 2shot_push,graph mode:TP2 4MB 19.9 → 9.1 µs 即 −54%,TP4 4MB 80.2 → 30.6 即 −62%)。gfx1250 的 cutoffs 是带着 clamp 选的,多个已偏离 crossover;#49955 也没重调 TP16。

## 优化点

- `custom_all_reduce.cuh`:删掉 clamp;unrolled 2shot_push 放开到 world size 16(原 ≤8),新增 `enough_warps` 检查(unrolled kernel 给每个 peer 一个 warp,块内 warp 数不足的 grid 走 per-vector kernel)。
- `two_shot_push_unrolled.cuh` 阈值:TP4 从 2 MB(原 8 MB;无 clamp 时 2 MB 起赢:背靠背 15%、late rank 10%);TP16 全尺寸(64 KB-64 MB 比 per-vector kernel 快 19-48%);TP2/TP3 保持 8 MB;TP5-TP8 不变。
- `dpx_pair_push.cuh`:新增 `pair_push_unrolled_kernel`(hier pair reduce-scatter/all-gather):warp 取整 hunk、先发全部 load 再用、批量 system-scope store;late peer 逐 hunk 重 poll;tail 走 per-vector。2 MB 及以上启用,以下 per-vector kernel 快 7-19%。
- `configs/custom_all_reduce_v2.py`:TP16(两表)hier_2shot_push 从 2 MB 起(256 KB-2 MB 走 unrolled 2shot_push;服务中每层两个 all-reduce 行为不同——attention 后的几 µs 内到达、MoE 后的因 per-socket DPM 与 expert 负载不均到达偏斜 100-1000 µs——各加一个后 512 KB 每层省 2.5 µs、1 MB 省 3 µs、2 MB 省 8 µs);dev_kernarg 表:TP2/TP4 1shot_push 到 1 MB;TP4 2shot_push 到 16 MB;TP8 hier_2shot_push 8-16 MB(仅当每加速器两个 DPX 分区都在组内;+64 MB hier planes per rank);TP16 hier_ring_simple 从 32 MB(不再建 ring_simple plane,TP16 每 rank 省约 96 MB:288 MB hier ring plane 代替 384 MB ring plane)。

## 性能声明

(数字全部照抄 PR 正文,未独立核实。环境:MI455X(gfx1250, DPX),`rocm/sgl-dev:v0.5.23-rocm101-mi45x-20261106`,`HIP_FORCE_DEV_KERNARG=1`;table dispatch,graph mode,bf16,µs per call,median of 3,max over ranks,每树 2 个交错 run;"Late" = 一个 rank 每次调用前 sleep 约 20 µs。)

TP2 2MB:6.5 → 5.9(−10%);TP2 4MB:19.8 → 9.0(−55%);TP4 2MB:22.4 → 17.7(−21%);TP4 4MB:80.5 → 25.6(−68%);TP6 512KB:18.5 → 14.3(−23%);TP6 1MB:27.9 → 17.7(−36%);TP8 12MB:75.1 → 65.7(−12%);TP8 16MB:92.8 → 79.9(−14%);TP16(2 hosts)512KB:22.1 → 22.6(+2%),1MB:27.1 → 23.7(−12%),2MB:28.0 → 25.8(−8%),64MB:408.5 → 291.1(−29%),128MB:534.3 → 446.0(−17%),256MB:843.0 → 760.5(−10%),512MB:1513.6 → 1407.9(−7%)。TP2/TP4/TP8 对比的是带 #50660 的 main(TP4 从 8 MB、TP8 从 20 MB 起两树都跑 hier_ring_simple);TP16 行用默认 16 MB communicator cap。`HIP_FORCE_DEV_KERNARG=0` 下 clamp 移除带来相近变化,无尺寸变慢。

## 验证与回归风险

无数值变化:每条路径仍按相同 rank 顺序求和,以下检查全部 exact。`test_custom_all_reduce.py --num-gpu 2,4,6,8` 配 `SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1`:TP4/6/8 共 599 passed;TP2 443 passed(hier 与 hier ring 用例跳过)。`test_custom_all_reduce_v2_capability.py` 与 `test_gfx1250_fabric.py`:117 passed。TP16 跨两主机:每个被测 provider(表 dispatch 默认/不设 cap、forced 1shot/2shot(含单 block grid)、hier、ring、hier ring、RCCL)在计时前对随机 per-rank 整数做 exact 检查(eager 与 CUDA graph),0 failures。新增 `test_hier_2shot_push`(graph 与 eager,bf16/fp16/fp32,2 MB 两侧的 pair chunks 加 partial hunk,对 NCCL exact;仅当组内含每加速器两个 DPX 分区才运行)。CI 状态:Base 通过,Extra 与 AMD ROCm 10 失败。回归风险:(1) 表项重排改变既有 TP 组合的 dispatch,依赖强制路径的用户需重验;(2) TP8 hier 范围带来每 rank +64 MB hier planes 内存;(3) 不能对半切成 16 B 向量的尺寸回退 NCCL(≤16 MB)或 hier_ring_simple;(4) unrolled kernel 的 enough_warps 分支与 TP16 单 block grid 行为。

## 关联

#49205(hier_2shot_push)、#49955(ring_simple 与 unrolled 2shot_push,本批次)、#50660(hier_ring_simple,本批次)——本 PR 明确在其之上移除 clamp 并重调两套表。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
