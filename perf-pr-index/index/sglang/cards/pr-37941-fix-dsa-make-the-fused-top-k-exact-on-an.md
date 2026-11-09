# PR #37941: [Fix] dsa: Make the fused top-k exact on an overflowing threshold bin

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xiaofei-zheng |
| 合并 | 2026-10-15 |
| 分类 | Sampler / Detokenization(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/37941 |

## 一句话结论

修复 DSA indexer fused top-k 在 threshold bin 溢出 2048 候选缓冲时静默返回错误 top-k 的问题,热路径仅 -1.7% 开销。

## 动机

fused top-k(v2,DSA/DeepSeek V4 路径)用 fp16 cast 的高位做粗直方图,一个 coarse bin 跨四分之一 binade;collect pass 最多暂存 2048 个 threshold-bin 候选(kMaxNumTie)。当 bin 更大时,`handle_tie` 对"先到先得"的子集排名,静默返回错误的 top-k 集合。在 MI355X 上 profile GLM-5.2 decode 发现:全部 8 个 rank 约 0.04% 的行进入该路径,观测到的最大 bin 为 4139(超过 2048 上限),且这些行无一 bit 相同。#35257 从 CUDA 侧独立报告了同一缺陷。

## 优化点

当 threshold bin 溢出暂存缓冲时走新增的 `refine_ties` 冷路径:用三轮 exact-key radix(12/12/8 bit)把粗 bin 收窄到单个 bucket,再对整行做最后一次扫描,发射 bucket 之上的候选并把 bucket 内元素重新暂存给既有 tie handler;refinement 直方图复用(overlay)已"死亡"的到达序暂存缓冲,LDS 占用不变。冷扫描用独立的 reversed-lane 加载路径,避免与快路径共享 load-address 状态(寄存器),恢复 review 中报告的大部分 CUDA 回归。同时把 fp16-bin 边界换算替换为精确 fp32 边界:`coarse_bin_bounds` 返回 {lo, hi},当 bin 上沿为 +0.0 时推进到最小正 subnormal,避免 -0.0 被误判为高于 bin。#39931 的 ROCm long-row split 路径获得同样 refinement(每 block 发布 uncapped bin count,最后一个 block 对整行 refine)。快路径仅新增一次溢出比较;ROCm `__launch_bounds__` 声明 4 waves/SIMD 物理下限,CUDA 拼写不变。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh: TopKSplit.finish_ties 溢出分支走 refine_ties;count_eq 改记 uncapped
python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh: refine_ties/refine_find_threshold 新增;coarse_bin_bounds 精确 fp32 边界;Register/Streaming 路径接入
test/registered/kernels/ops/attention/test_topk_v2.py: 新增 6 种窄分布 exactness 测试(max_permit_error=0)
```

```cpp
SGL_DEVICE static void finish_ties(const TopKProblem& problem, ...) {
  ...
  if (smem->total_eq > kMaxNumTie) [[unlikely]] {
    // The global list holds an arrival-order subset. refine_ties emits after
    // the slots every rank already took, so seed its counter with their total.
    if (tx == 0) smem->count_gt = smem->total_gt;
    __syncthreads();
    refine_ties(problem, smem, smem->v_lo, smem->v_hi, smem->total_eq);
    ...
  }
```

## 性能声明

MI355X(gfx950),单卡、每 arm 五次配对运行、warmup 后 200 次迭代:

- `randn` 热路径:6x8192 为 5.430 µs → 5.338 µs(-1.7%);4x32768 为 12.270 → 12.188 µs(-0.7%)。
- 溢出路径变慢(是"main 用错误答案跳过的精确计算"的成本):`narrow_bin` 6x8192 +24.5%、4x32768 +68.7%;`two_values`(需 32-bit refinement)+203.5% / +224.3%。serving 流量中约 0.04% 的行进入该路径。
- 相对本 PR 上一版实现,新扫描在 `narrow_bin` 上快 2.6-5.3%;无溢出时 split 路径不变(batch 1-8、64K-128K 为 15.5-20.1 µs)。
- 本地 JIT 干净构建 10.50 s vs main 10.02 s(+4.8%);gfx950 上 VGPR 63 → 71、零 spill、LDS 不变(26928 B)。CUDA 侧寄存器压力与 B200 性能未测(作者无 NVIDIA 硬件),需 CI/reviewer 测量。

## 验证与回归风险

新增 exactness 测试 `test_topk_v2_single_coarse_bin_is_exact`:register/streaming/split(4x65536)形状 x 六种分布(narrow、narrow_bin、tiny、two_values、signed_zero、all_equal 对照),`max_permit_error=0`;CUDA 上跳过 >24576 行(会路由到尚未 refine 的 cluster 路径)。修复前 10 个溢出非 bit-identical 用例全失败、signed-zero 用例在 pre-review refinement 上也失败;合并树上 330 例全过。已知缺口:CUDA-only 的 `TopKCluster` 路径仍会截断溢出 bin(refine 需要 cluster 级直方图与发射计数,本次未做)。风险:CUDA 侧性能未在本机验证;JIT 编译时间 +4.8%。

## 关联

#35257(CUDA 侧独立报告的同一缺陷,引用未关闭);#39931(ROCm long-row split 路径,本 PR 在其上叠加 refinement)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
