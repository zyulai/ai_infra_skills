# PR #62698: [Perf][MRV2] Parallelize staged writes across 1024-element chunks

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-10-25 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | ready, mrv2, kv-cache-manager |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/62698 |

## 一句话结论

`_apply_write_kernel` 改为每 1024 元素 chunk 一个 program，GB200 上 128K token 长写入快 31x、855K 快 120x。

## 动机

PR 正文 Overview 说明：`_apply_write_kernel`（MRV2 `StagedWriteTensor` / `FusedStagedWriter`）原本每次 staged write 启动一个 program，并在单个 SM 上循环遍历 1024 元素 chunks，把长 prompt 写入 `all_token_ids` 行时成为瓶颈。DEP decode 中每个 admission 在 admitting rank 的 forward 前执行该拷贝，使该 rank 成为该 step 的 straggler。此外 `BlockPool.get_new_blocks` 在未启用 prefix caching 且无 metrics collector 时仍有逐块检查开销。

## 优化点

改为一个 chunk 一个 program：grid dim 1 设为 `cdiv(max_write_len, 1024)`，超出较短写入末尾的 program 全掩码退出；`_apply_write_kernel` 内删除 `for i in range(0, content_len, BLOCK_SIZE)` 循环，改用 `tl.program_id(1)` 定位 chunk。每个 `StagedWriteTensor` 在 stage 时记录自身最长写入 `_staged_write_max_len`，使 launch 只增加 O(1) host 工作（正文称在 apply 时由 `cu_lens` 求 max 会在 Grace 上每 1024 次写入多花约 40 us）。`FusedStagedWriter` 同步聚合 `max_len`。写入字节数不变、不新增同步。`get_new_blocks`：无 caching/无 metrics 时因 free queue 弹出的块 `ref_cnt == 0`，直接置 `ref_cnt = 1`，跳过逐块检查；caching 与 metrics 路径不变。

## 关键改动

```text
vllm/v1/worker/gpu/buffer_utils.py: _apply_write_kernel 改为二维 grid + program_id(1) 分 chunk，新增 _WRITE_BLOCK_SIZE、_staged_write_max_len、_num_write_chunks
tests/v1/worker/test_gpu_block_table.py: 新增 test_block_tables_fused_write_spans_multiple_chunks 覆盖跨 chunk 的融合写入
```

```text
-        _apply_write_kernel[(n,)](
+        _apply_write_kernel[(n, _num_write_chunks(self._staged_write_max_len))](
...
+def _num_write_chunks(max_len: int) -> int:
+    return max(triton.cdiv(max_len, _WRITE_BLOCK_SIZE), 1)
...
-    for i in range(0, content_len, BLOCK_SIZE):
-        block = i + tl.arange(0, BLOCK_SIZE)
+    block = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

## 性能声明

Kernel microbenchmark（GB200，FlashInfer CUPTI，int32，旧 kernel 拷入脚本、输出比对相等后计时）：1 x 128K 76.32 → 2.46 us（31.0x）、1 x 855K 443.42 → 3.71 us（119.5x）、1 x 32K 20.94 → 2.18 us（9.6x）、1 x 8K 7.04 → 2.24 us（3.1x）、256 x 1 token 2.82 → 2.30 us（1.2x）；正文 Claims 概括为 "Long staged writes are 31x faster at 128K tokens (76 -> 2.5 us) and 120x faster at 855K tokens (443 -> 3.7 us) on GB200"。Serving context（DeepSeek V4.1 Flash，P/D decode，DEP，128K prompts，GB200）：DEP32 BS64 时 admitting rank 在 98 步中有 90 步最后到达首个 MoE、落后中位数 252 us（无 admission 的 step 为 46 us），改动后 DEP8 BS64 下落后 73 us（无 admission 为 63 us）；正文称 admission 由约 200 us rank skew 降为约 10 us。BlockPool 微基准（4000 blocks，enable_caching=False，无 metrics，GB200 Grace，best of 5 x 200）：`get_new_blocks` + `free_blocks` 616 → 528 us。

## 验证与回归风险

Tests（GB200）：`pytest tests/kernels/core/test_uva.py tests/v1/worker/test_gpu_block_table.py tests/v1/core/test_prefix_caching.py tests/v1/core/test_single_type_kv_cache_manager.py` 结果 "266 passed, 4 skipped"；已有 `test_staged_write_inflight` 覆盖多 chunk/混合长度写入，新增 `test_block_tables_fused_write_spans_multiple_chunks`。正文称 decode steps、block tables 等小写入不回退；权衡点是长写入与大量短写入混合时短写入 program 空转（1 x 128K + 63 x 1：5.94 us 对比单独 1 x 128K 的 2.5 us），仍远低于旧的 75 us。caching 与 metrics 路径保持不变。

## 关联

正文 "Duplicate check" 明确提到 #62018（把多次 staged-write 调用合并为更少 launch）、#51334 与 #59908（降低 host 侧 staging 开销），并说明它们与本 PR 互补、均不改变单个长写入的 kernel 并行方式。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
