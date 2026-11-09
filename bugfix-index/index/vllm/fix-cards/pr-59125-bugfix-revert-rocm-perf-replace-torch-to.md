# PR #59125: [Bugfix] Revert "[ROCm][Perf] Replace torch.topk in DSA candidate block selection"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59125](https://github.com/vllm-project/vllm/pull/59125) |
| 作者 | Fangzhou-Ai |
| 合并日期 | 2026-09-29 |
| labels | bug, rocm, ready |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#58208 的 Triton topk 替换在特定 shape 下有潜在 bug 且将被更快的 #58671 取代,整体回退恢复 torch.topk 路径。

## 问题现象
PR #58208 用 Triton kernel 替换 DSA candidate block selection 中的 torch.topk 后,在部分场景下存在正确性隐患。正文未给出具体报错或复现日志,仅以「某些特定 shape 下存在潜在 bug」概括;该替换同时引入了对后续优化路线的冗余。

## 根因
正文披露的两点原因:
1. 该替换在某些特定 shape 下存在潜在 bug(正文未披露具体 shape 与故障机制);
2. 该路径将被 #58671 引入的快得多的新路径取代,没有必要再继续改进这里的 kernel。
(由 diff 推断:被回退的 `_rocm_select_topk` 通过位打包构造保序 int64 key 做 top-K,对 NaN、平局、padding 等边界有复杂处理,这些正是易出错的边界。)

## 修复方案
回退 `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py` 中 #58208 的全部改动:删除 `_pack_key`、`_scan_range_topk`、`_live_blocks`、`_active_chunks`、`_write_candidates`、`_row_topk_kernel`、`_row_topk_partial_kernel`、`_row_topk_merge_kernel` 等 Triton kernel 及 `_rocm_select_topk` 调度函数,并移除 `select_candidate_blocks` 中的 ROCm 分支,恢复为统一走 `scores.topk(...)` + `_store_candidates_kernel` 的原有路径。

关键 diff 摘录(恢复后的选择逻辑):
```python
# Keep the existing top-k tie behavior.
top = scores.topk(min(topk_blocks, nblocks), dim=-1)
_store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
```

## 影响与回归风险
- 回归面:`select_candidate_blocks` 的 ROCm 执行路径;回退后 ROCm DSA 场景恢复 torch.topk 实现,失去 #58208 宣称的性能收益,直到 #58671 的新路径落地;CUDA 路径不受影响。

## 关联
- [PR #58208](https://github.com/vllm-project/vllm/pull/58208)
- [PR #58671](https://github.com/vllm-project/vllm/pull/58671)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
