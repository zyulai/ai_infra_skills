# PR #51210: [ModelRunner V2] Minor indexing optimizations

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2026-08-06 |
| 分类 | Other（tier: major） |
| Label | ready, mrv2 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/51210 |

## 一句话结论

ModelRunner V2 用 `__getitem__` 替 `get`、索引 dtype 改 `np.intp` / `torch.int64`，索引构造约快 6%。

## 动机

ModelRunner V2 的索引构造路径此前用 `dict.get` 配合 `np.fromiter(map(...))`，并以 `np.int32` / `torch.int32` 作为索引类型，后者在每次使用时会带来 copy / type-conversion 开销。本 PR 针对这些 CPU 侧索引构造做小幅优化，思路来自 PR #50532。

## 优化点

用 `dict.__getitem__` 取代 `dict.get` 作为 `np.fromiter(map(...))` 的映射函数（PR 标注约 6% faster）；索引类型改用 `np.intp`（numpy）与 `torch.int64`（gpu tensor index mapping），避免每次使用时的 copy / type-conversion。具体：`input_batch.py` 的 `make_dummy` 把 `idx_mapping_np` 的 `np.arange` dtype 由 `np.int32` 改为 `np.intp`、`idx_mapping` 的 `torch.arange` dtype 由 `torch.int32` 改为 `torch.int64`；`model_runner.py` 的 `prepare_inputs` 把 `num_tokens_per_req.get` / `req_id_to_index.get` 改为 `__getitem__` 并把 `np.fromiter` 的 dtype 由 `np.int32` 改为 `np.intp`；`pcp_manager.py` 的 `partition_batch` 把 `np.fromiter` 的 dtype 由 `np.int32` 改为 `np.intp`。

## 关键改动

```text
vllm/v1/worker/gpu/model_runner.py: prepare_inputs 改用 __getitem__ 映射，并把 np.fromiter 的索引 dtype 由 np.int32 改为 np.intp
vllm/v1/worker/gpu/input_batch.py: make_dummy 的 idx_mapping_np 改 np.intp、idx_mapping 改 torch.int64
vllm/v1/worker/gpu/pcp_manager.py: partition_batch 的 local_to_global_batch_req_idx_np 的 np.fromiter dtype 改 np.intp
```

## 性能声明

PR 正文仅在优化项旁标注 `Use __getitem__ rather than get with np.fromiter(map(...))` 约 6% faster（~6% faster），未给出 benchmark 表格或测试条件；PR 未注明测试条件（模型 / 硬件 / 精度 / workload）。正文无其他定量性能数据。

## 验证与回归风险

PR 正文未列出 Test Plan、单测或 Test Result（正文仅为三条优化说明与来源链接）。回归风险：改动集中在 ModelRunner V2 的 CPU 侧索引构造与 dtype（numpy `np.intp`、GPU tensor `torch.int64`），涉及 `prepare_inputs`、`make_dummy`、`partition_batch` 的 idx_mapping 路径；正文未给出受影响的模型、硬件、精度或已知回归信息。

## 关联

#50532（PR 正文明确 "Idea came from this PR" 并给出该链接）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
