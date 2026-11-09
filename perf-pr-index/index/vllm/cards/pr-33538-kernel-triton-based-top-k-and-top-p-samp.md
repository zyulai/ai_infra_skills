# PR #33538: [Kernel] Triton-based Top-k and Top-p sampler kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cakeng |
| 合并 | 2026-02-17 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用,且注明 4 个文件因超长未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/33538 |

## 一句话结论

用基于 Qrita 论文 pivot 截断+选择的 Triton kernel 替换采样前 top-k/top-p 的 PyTorch sort 路径,大批量下相对 PyTorch 最高提速约 30x。

## 动机

vLLM v1 采样前的 `apply_top_k_top_p` 走 PyTorch 实现:top-p 要对整个 vocab sort,top-k-only 快路径还需 GPU→CPU sync(对 async scheduling 不利)。此为 #25824 的重开版本,正确性与基准脚本取自 @njhill 的 #32558,目标是用单/少量 Triton kernel 完成 top-k/top-p 过滤。

## 优化点

机制(基于论文 "Qrita: High-performance Top-k and Top-p Algorithm for GPUs using Pivot-based Truncation and Selection",arXiv 2602.01518):零次 pass 从采样块估计均值/方差,用正态分位数表(`PERCENTILE_TO_STD_TABLE`/`NORMAL_CDF_TO_SIGMA_TABLE`)推出 outlier pivot,把少量离群 logit gather 进 `[num_program, vocab_size]` 形状 buffer(约 ~80MiB 额外显存)以缩小搜索空间;随后对 outlier 子集做三元搜索(ternary search)找 top-k pivot,终止条件 `k_pivots_num >= k 且 k_pivots_num - num_min_larger < k`,并处理重复 logit;top-p 在 top-k 结果上继续搜索,终止条件 `p_over_pivots_sum >= p 且 p_over_pivots_sum - (p_over_pivots_min * num_p_over_pivots_min) < p`,作者称比 #32558 更精确。调度上 `apply_top_k_top_p` 在 HAS_TRITON 且 batch >= 8 时走 Triton,小 batch 保留 PyTorch 路径;`forward_cpu` 显式走 pytorch 实现(允许 CPU sync),`apply_top_k_only` 注明含 GPU→CPU sync 的代价。

## 关键改动

```text
vllm/v1/sample/ops/topk_topp_triton.py: 新文件 1039 行,_topk_topp_kernel(TOPK/TOPP constexpr 组合)
vllm/v1/sample/ops/topk_topp_sampler.py: apply_top_k_top_p 按 batch 分派 Triton/PyTorch
vllm/utils/math_utils.py: power-of-2 辅助函数简化(顺带)
```

## 性能声明

PR 正文大表(batch 1-1024,vocab 32768/131072,scenario × Ops%,Triton vs #32558 vs PyTorch,单位 ms;硬件未注明):要点——相对 PyTorch,topk_whole 在 batch=48/131072 达 30.61x、batch=256/131072 topk_partial 41.46x;topk_topp_whole 普遍 3-21x;topk_whole@batch1/32768 为 1.55x(0.065 vs 0.101 ms)。相对 #32558:除 top-p-only 场景整体更快(如 batch1 topk_topp_whole 13.02x),但 topp_whole 大 batch 更慢(batch=1024/131072 时 0.23x:13.691 vs 3.172 ms)。显存:Triton 略高于 #32558(batch1024/131072:1.56 GB vs 1.50 GB),远低于 PyTorch(5.00 GB)。

## 验证与回归风险

PR 正文称通过全部正确性测试(测试文件在未收录的 4 个 diff 文件中,内容不可见)。风险面:batch < 8 仍走 PyTorch(保持小 batch 延迟);stochastic cutoff 是概率性截断,依赖统计假设,极端分布(全 -inf、重复 logit、grammar mask)在 kernel 内有专门处理(排除 -inf、重复计数);~80MiB buffer 按设备缓存,长上下文大 batch 显存占用增加;HIP/CPU 路径不受影响(forward_hip/forward_cpu 保留)。

## 关联

同区域:#25824(本 PR 的前身)、#32558(njhill 的实现,基准与正确性来源,本 PR 在多数场景下更快)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
