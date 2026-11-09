# PR #41947: [AMD][V4.1][*/N] Route low-ratio indexer and candidate-block top-k through top-k v2

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu |
| 合并 | 2026-10-01 |
| 分类 | Attention / Kernel(tier: major) |
| Label | jit-kernel |
| Diff | pack diffstat 缺失;diff 为 candidate_blocks_hip.py 与 low_ratio_backend_hip.py 两个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/41947 |

## 一句话结论

DSV4.1 low-ratio indexer 在 ROCm 上弃用两个 AOT top-k kernel、改走 top-k v2(去掉排序),kernel 快 1.4–2.6x,8k1k 端到端吞吐最高 +7.7%。

## 动机

ROCm 上 DSV4.1 low-ratio indexer 的 top-k 仍在用旧 AOT kernel。MI355X 上 v2 在 decode shapes 快 1.4–2.6x 且选出完全相同的 indices。被替换的两个 AOT kernel:`deepseek_v4_topk_transform_512`(选 top 512 positions,既用于 full row 也用于 candidate blocks 内部)与 `fast_topk`(选 top 2048 candidate blocks)。AOT top-512 kernel 还会按 position 对每行排序,但消费该 selection 的 attention kernel 并不关心顺序:decode attention 读前 `lengths[t]` 个条目并跳过 −1;prefill(`combine_topk_swa_indices`)只要求未使用条目为 −1。CUDA 路径本来就 unsorted 传递 selection,因此本 PR 在 ROCm 上同样去掉排序。

## 优化点

`candidate_blocks_hip.py` 新增 `topk_transform_paged_hip()`,把 unsorted 的 `topk_transform_paged` 跑在 top-k v2 kernel 上(`topk_transform_paged_v2` + `plan_topk_v2`;plan 在 MI355X 上不写任何东西),并要求 score 行 16-byte 对齐(`stride % 4 == 0`)——`candidate_block_scores` 的输出分配相应把 rows 补齐到 16 字节再切片。`select_candidate_blocks_hip` 的 top-2048 candidate-block 选择从 AOT `torch.ops.sgl_kernel.fast_topk` 改为该函数;`topk_within_candidate_blocks_hip` 从 AOT `topk_transform_paged` 改为该函数。`low_ratio_backend_hip.py` 的 decode 与 extend 两条路径全部从 `topk_transform_paged_sorted` 切换过来,并删除 `sort_output=True` 参数;非 2048 的 k 仍走 `torch.topk` 回退。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/candidate_blocks_hip.py: 新增 topk_transform_paged_hip(v2、unsorted);candidate-block top-2048 改走 v2
python/sglang/srt/layers/attention/dsv4/low_ratio_backend_hip.py: decode/extend 的 top-k 全部切换到 v2 并去掉 sort_output
```

## 性能声明

Kernel level,MI355X,graph replay,µs(v2 含 `plan_topk_v2`,MI355X 上该 plan 不写):

Paged top-512(plain path 与 candidate blocks 内 top-k):rows/width 20/4096:13.3 → 5.0;80/8192:14.7 → 6.2;320/8192:18.3 → 10.6;80/16384:19.2 → 7.8;320/16384:24.4 → 14.5。Candidate-block 选择 top-2048:20/4096:8.5 → 4.6;80/8192:10.9 → 5.9;320/8192:20.3 → 7.7;80/16384:14.4 → 8.6;320/16384:29.2 → 13.5。

e2e(8k1k workload;PR 未注明 e2e 使用的硬件与模型):conc 1:TTFT 400.29 → 393.85(−1.6%)、TPOT 1.66 → 1.65(−0.6%)、Throughput 411.46 → 443.28(+7.7%);conc 4:TTFT −3.3%、TPOT −0.3%、Throughput 839.72 → 848.77(+1.1%);conc 16:TTFT 1062.83 → 976.84(−8.1%)、TPOT −3.1%、ITL 持平、Throughput +0.2%。

## 验证与回归风险

gsm8k 1319 题:0.902(PR 未给出 baseline 对照值);选择正确性的依据是 v2 与 AOT kernel "selects exactly the same indices"。风险:top-k v2 要求 16-byte 对齐的 score 行(新分配已保证,但调用方传入的 tensor 需满足);`fill_tail` 语义随 `use_kernel` 变化;e2e 吞吐增益在 conc 4/16 明显收敛。

## 关联

无(pack 未给出其他 PR 编号;CUDA 路径的 unsorted 行为为既有现状)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
