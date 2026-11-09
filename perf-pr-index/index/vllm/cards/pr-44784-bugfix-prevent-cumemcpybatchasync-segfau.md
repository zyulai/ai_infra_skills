# PR #44784: [Bugfix] Prevent cuMemcpyBatchAsync segfault with MTP and KV offloading

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JOSH1024 |
| 合并 | 2026-06-16 |
| 分类 | Speculative Decoding (tier: major) |
| Label | bug, ready, v1, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 "file(s) dropped from diff" 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/44784 |

## 一句话结论

修复 native KV offloading 与 MTP 在长上下文并用时的 cuMemcpyBatchAsync CUDA segfault。

## 动机

vLLM 在 `--kv-offloading-backend native` 与 MTP speculative decoding 一起用于长上下文时崩溃，报 `cuMemcpyBatchAsync` segfault。这是静默、难以定位的 CUDA 崩溃而非普通 Python 异常，需要定位根因并给出显式防线。正文未给出更多触发条件或影响范围描述。

## 优化点

根因：`OffloadingConnectorScheduler` 错误地把 EAGLE/MTP draft-attention groups 同时纳入 load 与 store 调度路径。这些 draft group 没有稳定 block hashes 且使用极小的 `gpu_block_size`，导致 `num_gpu_blocks = cdiv(420K, small_size) = 105022` 的巨大误算，产生越界 GPU block index，在 `swap_blocks_batch` 中触发静默 CUDA segfault。修复分三层：给 `GroupOffloadConfig` 增加 `is_eagle_group` 标志以便识别；调度器在所有 load/store 调度循环里完全跳过 eagle groups；并在 `gpu_worker.py` 增加 pre-DMA bounds check 作为纵深防御，使任何漏过的越界 block ID 抛清晰的 Python `RuntimeError` 而非静默 CUDA segfault。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: 增加 is_eagle_group 标志，并在 load/store 调度循环跳过 eagle groups
tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py: 新增 TestEagle 类，覆盖 _lookup 单测与集成测试、is_eagle_group 默认值、load/store 路径跳过与 bounds check
```

正文另提到 `gpu_worker.py` 的 pre-DMA bounds check，但该文件的 diff 未包含在素材中（pack 的 diff 仅保留上述两个文件）。

关键代码（store 路径跳过 eagle group，来自 scheduler.py diff 末尾）：

```diff
+                if group_config.is_eagle_group:
+                    num_blocks = max(0, num_blocks - 1)
+
                 start_block_idx = group_state.next_stored_block_idx
                 if num_blocks <= start_block_idx:
                     continue
```

## 性能声明

PR 正文未给出定量性能数据。文中的 `num_gpu_blocks = cdiv(420K, small_size) = 105022` 属于根因误算说明，非 benchmark 结果；`8 passing tests` 为回归测试数量。PR 未注明测试条件（模型/硬件/精度/workload 均未给出）。

## 验证与回归风险

作者用自定义独立回归测试套件本地验证（8 passing tests）：确认 `is_eagle_group` 正确填充且默认 `False`（保持向后兼容）；验证 load 路径 `update_state_after_alloc` 与 store 路径 `_build_store_jobs` 恰当地跳过 MTP groups 并返回 `group_sizes=0`、`block_indices=0`；验证纵深防御 bounds check 在触发 CUDA 调用前安全捕获越界 ID。作者在 Windows 上开发，依赖 GitHub CI 运行完整 Linux 套件，并说明使用过 AI 辅助定位根因与写补丁但逐行人工复核。可能受影响路径：native KV offloading + MTP/EAGLE 长上下文、offloading connector 的 load/store 调度与 DMA 边界。

## 关联

无（正文仅以 `Closes #44780` 关联一个 issue，未给出前序/后续 PR 编号线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
