# PR #42191: [Perf] Apply single-pass min_larger finding and binary search in Triton Top-p path.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cakeng |
| 合并 | 2026-06-03 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 仅含 `vllm/v1/sample/ops/topk_topp_triton.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/42191 |

## 一句话结论

Triton top-k/top-p kernel 的 Top-p 路径改用单 pass min_larger 统计加二分 pivot 搜索,Top-p 路径提速 25-40%。

## 动机

Top-p 路径此前用三分数(two pivots)搜索 p_pivot,每轮迭代需要两次缓冲区扫描:第一遍算 `p_pivots_sum` 与 `min_larger`,第二遍另扫一遍算 `num_min_larger`;且要维护两套 pivot 状态变量,寄存器压力大。#37225 已引入 tile 级合并的 min_larger 统计 helper(`_update_min_larger_stats`),可以在一次遍历中同时得出 sum、min_larger 和 num_min_larger,适合移植到 Top-p 路径。

## 优化点

三处机制改动。其一,Top-p 的 p_pivot 搜索从三分数(1/3、2/3 两个试探点)改为二分(单试探点取 0.5),每轮少算一套 pivot 的统计与比较分支,用省下的寄存器容纳 min_larger helper 带来的寄存器压力增长。其二,遍历缓冲区时把 `p_pivots_sum`、`min_larger`、`num_min_larger` 融合为单 pass:先算 `above_0 = probs_blk > p_pivot_0`,再经 `_update_min_larger_stats` tile 级合并,消除了原来的第二遍完整扫描。其三,顺带修两个 bug:Top-p logit 路径的 `duplicate_logit` 计算由 `+ max_logit` 改为 `+ max_sample`;重复 logit 的保留掩码由 `duplicate_count <= num_duplicate_logit` 改为 `duplicate_count <= num_keep`(按超出的重复数裁剪)。Top-k pivot 搜索中终止时也补回 `min_larger`/`num_min_larger` 的赋值。

## 关键改动

```text
vllm/v1/sample/ops/topk_topp_triton.py: _topk_topp_kernel 的 Top-p 搜索
  改单 pass + 二分;duplicate_logit 与 keep mask 修复
```

关键片段(单 pass 统计):

```python
above_0 = probs_blk > p_pivot_0
p_pivots_sum_0 += tl.sum(probs_blk * above_0)
min_larger_0, num_min_larger_0 = _update_min_larger_stats(
    probs_blk, above_0, min_larger_0, num_min_larger_0, 1.0,
)
```

## 性能声明

H100 上 `benchmarks/benchmark_topk_topp.py`,vocab 32768 与 131072,batch 1-2048,各场景(PyTorch/OrigTri/CurrTri 三方对比,单位 ms):Top-p 路径 25-40% 提速。例:vocab 32768,batch 512 topp_whole 0.791 → 0.636(1.24x)、topp_partial 0.400 → 0.321(1.25x);vocab 131072,batch 2048 topp_whole 5.655 → 4.372(1.29x)、topp_partial 2.864 → 2.210(1.30x)、topp_partial batch512 0.746 → 0.579(1.29x)。topk 路径与 topk_topp 组合路径基本持平(约 1.00x-1.2x),个别 mixed_partial 略有回退(如 batch 512/vocab 32768 的 0.95x)。显存占用前后一致。

## 验证与回归风险

正确性用 `tests/v1/sample/test_topk_topp_sampler.py`(作者:全部通过)。风险:两处行为修复(duplicate_logit 的基准从 max_logit 改为 max_sample、keep mask 从 num_duplicate_logit 改为 num_keep)会改变存在重复 logit 时的采样边界结果,属于正确性修正但可能改变既有输出;搜索从三分改二分理论上迭代轮数上限不变(18 轮 + 1e-9 收敛阈值)。

## 关联

依赖/复用 #37225 的 `_update_min_larger_stats` helper。同分类 kernel 优化:#42235、#55059、#60084、#27284。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
