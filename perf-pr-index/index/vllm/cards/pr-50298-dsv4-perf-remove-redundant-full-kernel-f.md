# PR #50298: [DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-07-30 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未见「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/50298 |

## 一句话结论

给 combine_topk_swa_indices 传 out tensor 省掉 torch.full，B300 上该内核 1.88x（GPU）/1.86x（Wall）。

## 动机

本 PR 是 issue #45861 的一部分。正文 Purpose 只有一句："Passing an out tensor to avoid additional `torch.full` kernel call"——原实现每次调用都要用 `torch.full` 分配并填充一个 int32 索引缓冲区，带来一次额外 kernel 启动，属于可消除的冗余开销。

## 优化点

给 `combine_topk_swa_indices` 增加可选参数 `out: tuple[torch.Tensor, torch.Tensor] | None = None`。调用方若传入预分配的 `(combined_indices_out, combined_lens_out)`，则直接复用该缓冲区，跳过 `torch.full` 创建 `combined_indices` 与 `torch.empty` 创建 `combined_lens` 的过程；未传 `out` 时保持原有分配逻辑不变。改动从内核工具函数延伸到 DSv4 的 FlashMLA prefill 调用点，由调用方预先分配 workspace 并按轮复用，机制上是用输出缓冲复用换取一次 kernel launch 与分配的消除。

## 关键改动

- `vllm/models/deepseek_v4/common/ops/cache_utils.py`：`combine_topk_swa_indices` 新增 `out` 参数，`out is None` 时走原分配，否则直接解包复用。
- `vllm/models/deepseek_v4/nvidia/flashmla.py`：调用点改为传入 `out=(combined_indices_out, combined_lens_out)` 预分配缓冲。
- `tests/kernels/attention/test_flashmla_sparse.py`：对应单测调整（正文称 "Covered in unit tests"）。

```text
+    if out is None:
+        combined_indices = torch.full(
+            (num_tokens, combined_topk),
+            fill_value=-1,
+            dtype=torch.int32,
+            device=topk_indices.device,
+        )
+        combined_lens = torch.empty(
+            num_tokens, dtype=torch.int32, device=topk_indices.device
+        )
+    else:
+        combined_indices, combined_lens = out
```

## 性能声明

正文所附 AI 生成脚本输出（GPU: NVIDIA B300 SXM6 AC；DSv4 C128A prefill combine-index microbenchmark；Prompt tokens 8192、Maximum model len 1048576、C128A index width 8192、Combined width 8320、Output buffer 260.0 MiB；Timing 5 rounds x 20 calls；Correctness check passed）：before GPU latency 0.093 ms / Wall latency 0.094 ms，after GPU latency 0.049 ms / Wall latency 0.050 ms，"GPU: 1.88x speedup, 46.8% lower latency"、"Wall: 1.86x speedup, 46.3% lower latency"，标题称 "1.88x kernel performance improvement"。脚本明确注明 "This is the combine-index micro-operation speedup, not full-model prefill speedup."，即仅该微操作加速，非整模型 prefill；模型与精度 PR 未注明测试条件。

## 验证与回归风险

验证：正文称 "Covered in unit tests"，microbenchmark 内含 correctness check（lengths 与 sampled valid entries，passed）。可能受影响路径：DSv4 C128A prefill 的 combine-index 与 FlashMLA sparse 调用、未传 `out` 的其它调用方（保持原行为）；复用缓冲区要求调用方保证缓冲生命周期与内容不被并发写坏，正文未列出具体回归测试文件名。

## 关联

正文标注 "Part of https://github.com/vllm-project/vllm/issues/45861"（issue 编号而非 PR 编号），未给出同分类其它 PR 编号或前序/后续 PR，按 PR 编号口径写「无」。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
