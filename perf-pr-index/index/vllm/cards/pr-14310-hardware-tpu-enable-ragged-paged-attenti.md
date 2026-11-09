# PR #14310: [Hardware][TPU]Enable ragged paged attention kernel and resolve recompilation issue

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yaochengji |
| 合并 | 2025-03-06 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, ci/build, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/14310 |

## 一句话结论

TPU 上启用 ragged paged attention kernel，并解决重编译问题。

## 动机

正文以 4 条列出改动：enable ragged paged attention kernel、resolve recompilation issue、把 KV cache 布局从 `(num_kv_heads, num_blocks, block_size, head_size)` 改为 `(num_blocks, block_size, num_kv_heads, head_size)`、更新 torch xla nightly wheel。正文未说明更细的瓶颈背景与收益，故具体动机 PR 未注明。

## 优化点

`vllm/v1/attention/backends/pallas.py` 中 `NUM_QUERIES_PER_BLOCK` 由 16 改为 32，`get_kv_cache_shape` 返回新布局；`__init__` 依据 `torch_xla.tpu.version()` 要求 TPU >= 4，v4 设 `vmem_limit_bytes=16MB`、其余 64MB；`forward` 移除 `query = query * self.scale` 与 `use_kernel = False` 的引用实现开关，改为调用 `torch.ops.xla.ragged_paged_attention` 时传 `vmem_limit_bytes`、`use_kernel=True`、`sm_scale=self.scale`。`write_to_kv_cache` 改为 `flatten(0, 1)` 并 flatten `slot_mapping`。`tpu_model_runner.py` 的 padding 从按 `NUM_QUERIES_PER_BLOCK` 的 `_get_padded_number` 改为 `_get_padded_token_len`（≤16 返回 16，否则取 2 的幂），metadata 按 `max_num_reqs` 构造、`num_seqs` 改为 int32 tensor，并移除 forward 里为旧 KV cache 布局重算 `slot_mapping` 的大段逻辑与若干 `mark_dynamic`（对应 resolve recompilation）。`requirements-tpu.txt` 把 torch / torch_xla nightly wheel 从 dev20250227 升到 dev20250306。`pallas.py` 的 diff 注释说明 `NUM_QUERIES_PER_BLOCK` 与 `NUM_KV_PAGES_PER_BLOCK` 是 paged attention Pallas kernel 的两个可调参数（后者仍为 128），并注明 "TPU v4's vmem capacity is 16MB"、TODO 后续 autotune 这三个参数；`_get_padded_token_len` 的实现为 `x <= 16` 时返回 16，否则 `1 << (x - 1).bit_length()`，即向上取到 2 的幂，配合按 `max_num_reqs` 固定形状来避免反复编译。

## 关键改动

```text
vllm/v1/attention/backends/pallas.py: 启用 ragged_paged_attention kernel（use_kernel=True、sm_scale、vmem_limit_bytes），KV cache 改新布局
vllm/v1/worker/tpu_model_runner.py: 按 max_num_reqs 构造 metadata、_get_padded_token_len、移除旧布局 slot_mapping 重算
requirements-tpu.txt: torch / torch_xla nightly wheel 由 dev20250227 升到 dev20250306
```

kernel 启用片段（摘自 pallas.py 的 diff）：

```diff
-            use_kernel=use_kernel,
-        )
+            vmem_limit_bytes=self.vmem_limit_bytes,
+            use_kernel=True,
+            sm_scale=self.scale)
```

## 性能声明

PR 正文未给出定量性能数据：正文只有 4 条改动清单，无 benchmark 数字，也没有模型 / 硬件 / 精度 / workload 等测试条件（可从路径与 label 看出面向 TPU，但正文未给出具体型号与 workload）。

## 验证与回归风险

PR 未注明 Test Plan / Test Result 与具体测试用例。回归风险：diff 中 `tpu_version < 4` 直接 `NotImplementedError("TPU version must be 4 or higher.")`；正文称 resolve recompilation issue（对应按 `max_num_reqs` 固定 metadata 形状、移除动态 `mark_dynamic`）；KV cache 布局变化同时影响 `write_to_kv_cache` 与 `tpu_model_runner`；nightly wheel 升级到 dev20250306。其余路径 PR 未注明。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
