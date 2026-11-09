# PR #6998: Fix cutlass MLA gets almost zero accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6998](https://github.com/sgl-project/sglang/pull/6998) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-06-09 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

cutlass MLA 开 CUDA graph 后 GSM8K 精度近乎归零;capture/replay 改用固定 padding 与预分配 workspace 修复。

## 问题现象

DeepSeek-V3-0324 + `--attention-backend cutlass_mla`(TP 8 + DP attention 8)下,GSM8K(`benchmark/gsm8k/bench_sglang.py --parallel 10000 --num-questions 1400`)精度异常:

- master + 开 CUDA graph:Accuracy 0.060,几乎为零;
- master + `--disable-cuda-graph`:Accuracy 0.910~0.920;
- 本 PR + 开 CUDA graph:Accuracy 0.906~0.917,恢复到与不开 CUDA graph 基本持平(略低);
- 参考 baseline + triton 后端:Accuracy 0.927~0.937。

正文还指出:non-cuda-graph 路径似乎仍有 bug,需另行排查。

## 根因

正文只说「Fix issues in cuda graph」,未披露具体根因。由 diff 可描述机制:capture 与 replay 时 `max_seqlen_pad` 都按当时 `seq_lens.max().item()` 动态计算,并据此重新分配 `cuda_graph_mla_workspace`、对 `cuda_graph_kv_indices` 按 `[:bs, :max_seqlen_pad]` 重新切片;CUDA graph 捕获时固化的 workspace 大小与 kv_indices 形状是按 capture 时刻的 batch 形状来的,replay 阶段实际 seqlen/bs 不同时会与图内记录不一致,导致注意力读到错误的元数据,精度崩掉(由 diff 推断)。

## 修复方案

改 `python/sglang/srt/layers/attention/cutlass_mla_backend.py`:

- `init_forward_metadata_capture_cuda_graph`:`max_seqlen_pad` 不再按 `seq_lens.max().item()` 计算,改为固定取 `self.cuda_graph_kv_indices.shape[1]`;删除按当前 bs/seqlen 重建 workspace 的代码,直接使用预分配的 `self.cuda_graph_mla_workspace`。

```diff
-                max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)
+                max_seqlen_pad = self.cuda_graph_kv_indices.shape[1]
```

- `init_forward_metadata_replay_cuda_graph`:删除 `seq_lens_cpu.max()` 计算、workspace 重新分配与 `forward_metadata` 字段的重建,仅保留 triton kernel 写 kv_indices。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 cutlass_mla 后端的 CUDA graph capture/replay 元数据路径;workspace 改为按图的最大形状预分配,显存占用可能略增;非 CUDA graph 路径与本改动无关(正文提示该路径可能另有 bug)。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
