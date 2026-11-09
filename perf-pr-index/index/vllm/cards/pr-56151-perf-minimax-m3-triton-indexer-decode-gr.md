# PR #56151: [Perf][MiniMax-M3] Triton indexer: decode grid retune + SM12.0 split-K

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @venkywonka |
| 合并 | 2026-09-30 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, ci/build, nvidia, minimax |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 `vllm/models/minimax_m3/common/ops/index_topk.py` 与 `tests/kernels/attention/test_minimax_m3.py` 两个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/56151 |

## 一句话结论

MiniMax-M3 Triton indexer decode grid 重调 + SM120 prefill split-K：高 batch decode step p50 49.2ms→39.6ms，短 extend prefill 迭代 263ms→200ms。

## 动机

MiniMax-M3 的 Triton indexer（`index_topk.py`）decode 路径的 launch grid 参数（score 512 chunks、top-k 64 chunks）在高 batch 下偏小，SM 占用不足导致 decode step 时间明显劣于参考栈；prefill 侧短 chunked-prefill extend 时 launch grid 很小，K 维串行循环利用率低。本 PR 从 #49229 折叠入 grid 重调部分（原提交 Zijing Liu、Kyle Liang），并给 SM12.0 加 prefill split-K。PR 曾因 concurrency-12 点位无收益被关闭，重开后收窄：FP8 index-cache 与 operand cast 移入 #59081，fused decode top-k 与 router 改动被放弃。

## 优化点

两处机制改动。其一，decode launch grid 重调：`TARGET_GRID` 512→4096（decode index score）、`TOPK_TARGET_GRID` 64→512（decode top-k），对使用 Triton indexer 的所有 CUDA GPU 生效（ROCm 走独立的 `amd/ops/index_topk.py` 不受影响）；chunk 数仍只依赖 shape 常量，CUDA graph 安全，≤2 个 decode 请求时 score grid 不变（两值都触及 256-chunk 上限）。其二，SM12.0 专属 prefill split-K：`_index_block_score_kernel` 增加 `USE_SPLIT_K` 维度，`split_k = min(max_block, 48 // (n_q_tiles * batch * heads))`，把 K 维 block 循环切到多个 CTA；每个 CTA 写不相交的 per-block maxima 区间，无需归约；超出 causal window 的 split 提前退出。非 SM12.0 GPU 强制 `split_k = 1` 保持原 launch。

## 关键改动

```text
vllm/models/minimax_m3/common/ops/index_topk.py: decode TARGET_GRID/TOPK_TARGET_GRID 重调；prefill score kernel 增加 USE_SPLIT_K 与 SM120 分派
tests/kernels/attention/test_minimax_m3.py: prefill correctness 测试参数化 prefix_len 与 force_split_k（任何 CUDA GPU 可强制 split 路径）
```

```python
split_k = max(1, min(max_block,
    SCORE_TARGET_GRID // max(1, n_q_tiles * batch * num_idx_heads)))
if not (current_platform.is_cuda()
        and current_platform.is_device_capability((12, 0))):
    split_k = 1
grid_score = (n_q_tiles, batch * num_idx_heads, split_k)
```

## 性能声明

测试条件：MiniMax-M3-NVFP4 + EAGLE3，8x RTX PRO 6000（SM120），InferenceMAX recipe，c10 TP8。

- Grid retune：去掉该项 throughput 变 0.954x（即贡献 4.6%）；10 请求时 decode step p50 无该项 49.2ms、有该项 39.6ms（与参考栈 39.6ms 持平）。#49229 另报告过 B300 TP4 c24 下 +44%。
- Split-K：短 cached extends 每迭代 263ms→200ms（参考栈 192ms），来自对早期 fused-top-k+split-K 版本的 leave-one-out。
- 本 PR 整体（split-K 确认 run 已含 grid retune）：对 InferenceMAX 参考栈 0.985 tput / 1.004 interactivity；对早期 fused+split-K 版本 per-request TPOT 0.997 [0.975, 1.008]（n=1）。
- SM120 测试：95 passed / 0 failed / 58 skipped（main 为 92/0/58，差异仅 3 个新 prefill 用例）。

## 验证与回归风险

`test_prefill_index_topk_correctness` 参数化 `force_split_k`，通过 monkeypatch `is_device_capability` 在任意 CUDA GPU（含 L4 CI、不均匀 split、prefix_len=3600 尾块）上强制走 split 路径；`test_decode_index_topk_correctness` 覆盖新 decode grids（256 score chunks、最多 16 top-k chunks）全部 6 用例；4 个 prefill 用例均走 `USE_SPLIT_K=True`。风险面：grid 重调影响所有 CUDA GPU 上的 MiniMax-M3 Triton indexer decode（2 请求以下无变化）；split-K 仅限 SM12.0，其他 GPU `split_k=1` 不受影响；ROCm 路径未触碰。

## 关联

前序/来源：#49229（grid retune 折叠入本 PR）；#59081（FP8 index-cache 与 operand casts，取代 #47665，从本 PR 拆出）。模型基础：MiniMax-M3 支持（#45381，索引内）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
