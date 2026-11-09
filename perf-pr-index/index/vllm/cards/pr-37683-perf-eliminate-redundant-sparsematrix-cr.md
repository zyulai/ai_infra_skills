# PR #37683: [Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-03-20 |
| 分类 | Attention / Kernel（tier: major） |
| Label | performance, ready, gpt-oss |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 2 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/37683 |

## 一句话结论

避免 gpt_oss Triton MoE 重复构造 SparseMatrix（3 个 bitmatrix kernel 由 2 次降为 1 次），H200 / batch_size=1 e2e output tok/s 提升 4%。

## 动机

正文 Purpose：profiling 时发现 `_sum_bitmatrix_rows` + `_bitmatrix_metadata_compute_stage1` + `_bitmatrix_metadata_compute_stage2` 三个 kernel 被 launch 了两次。原因是创建 `SparseMatrix` 对象本身会 launch 这 3 个 kernel；`legacy_routing` 中的 `topk` 已经创建过一次 `SparseMatrix`，而 `legacy_routing_from_bitmatrix` 内部又再次创建，导致这 3 个 kernel 重复执行。

## 优化点

直接把 triton_kernels `topk` 返回的 `sparse_logits`（`SparseMatrix`）向下传递，避免再次重建 `SparseMatrix` 对象，从而消除冗余的 3 个 bitmatrix metadata kernel launch。为此新增独立的 `legacy_routing_from_sparsematrix` 函数，不改同样被其他路径使用的 `legacy_routing_from_bitmatrix`；`legacy_routing` 改为调用新函数并传整个 `sparse_logits`。diff 中新函数直接从 `sparse_logits.mask_metadata` 取 `row_sorted_indx` / `col_sorted_indx` / `col_sum` 构造 routing 数据。Profiling 对比：Main 三个 kernel launch 两次，PR 一次。

正文还给出冗余来源的具体代码位置：`legacy_routing` 中的 `topk`（gpt_oss_triton_kernels_moe.py L160）创建一次 `SparseMatrix`，`legacy_routing_from_bitmatrix`（L108）又在 L125 处再次创建；创建动作对应 triton_kernels 的 `SparseMatrix`（tensor.py L233）与 `make_bitmatrix_metadata`（bitmatrix.py L110）。新增的 `legacy_routing_from_sparsematrix` 直接取 `sparse_logits.mask_metadata.row_sorted_indx` / `col_sorted_indx` / `col_sum` 及 `sparse_logits.vals.flatten()[combine_indx]`，配合 `make_ragged_tensor_metadata` 构造 `RoutingData`、`GatherIndx`、`ScatterIndx`；`test_gpt_oss_triton_kernels.py` 相应增加用例。正文 Implementation details 还列出三点：直接传递 `topk` 返回的 `sparse_logits`、另建独立函数以免改动同样被其他地方使用的 `legacy_routing_from_bitmatrix`、在 `test_gpt_oss_triton_kernels.py` 中补测试；Serving benchmark 对 Main 与 PR 各在 concurrency=1 和 16 给出一组完整 serving result。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py: 新增 legacy_routing_from_sparsematrix，legacy_routing 改为直接传 sparse_logits
tests/kernels/moe/test_gpt_oss_triton_kernels.py: 增加 test_legacy_routing 相关用例
```

```diff
     sparse_logits = topk(logits, n_expts_act, apply_softmax=not sm_first)
-    return legacy_routing_from_bitmatrix(
-        sparse_logits.mask,
-        sparse_logits.vals,
-        sparse_logits.indx,
+    return legacy_routing_from_sparsematrix(
+        sparse_logits,
         logits.shape[-1],
         n_expts_act,
     )
```

## 性能声明

测试条件：H200、batch_size=1；`vllm serve /opt/dlami/nvme/models/gpt-oss-20b --tensor-parallel-size 1 --max-num-seqs 16 --no-enable-prefix-caching`；`vllm bench serve --dataset-name sharegpt --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 300 --num-warmups 50 --ignore-eos --temperature 0`（num-prompts / max-concurrency 按并发取值）。正文结论："e2e output token throughput improves 4% at batch_size=1 on H200."（perf claims 段同样标注 4%）。

| Metric | Main c=1 | PR c=1 | Main c=16 | PR c=16 |
|---|---|---|---|---|
| Output token throughput (tok/s) | 265.06 | 275.91 | 2306.46 | 2354.31 |
| Total token throughput (tok/s) | 494.76 | 515.03 | 4046.73 | 4130.68 |
| Mean TTFT (ms) | 18.03 | 17.70 | 63.83 | 72.87 |
| Mean TPOT (ms) | 3.72 | 3.58 | 6.74 | 6.57 |
| Median ITL (ms) | 3.73 | 3.58 | 6.57 | 6.38 |

## 验证与回归风险

验证：`pytest -s -v tests/kernels/moe/test_gpt_oss_triton_kernels.py::test_legacy_routing`（正文称 Unit test passed）；Profiling 截图对比；上述 serving benchmark；Accuracy Testing `OPENAI_API_KEY=EMPTY python3 -m gpt_oss.evals --model /opt/dlami/nvme/models/gpt-oss-20b --eval gpqa --n-threads 200 --reasoning-effort low`，Main score 0.5574494949494949、PR 0.5675505050505051。可能影响路径：gpt_oss Triton MoE 的 `legacy_routing` 路径；正文说明保留 `legacy_routing_from_bitmatrix` 供其他调用方使用，故新函数不影响该路径。PR 未注明其他回归风险。

## 关联

无（正文引用的均为源码/仓库文件链接，非 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
