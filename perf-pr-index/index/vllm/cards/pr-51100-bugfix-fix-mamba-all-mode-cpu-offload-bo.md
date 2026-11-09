# PR #51100: [Bugfix] Fix Mamba all-mode CPU offload boundary alignment

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jairitAge |
| 合并 | 2026-08-07 |
| 分类 | KV Cache / Memory \| tier: major |
| Label | bug, ready, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明（diff 完整呈现） |
| 链接 | https://github.com/vllm-project/vllm/pull/51100 |

## 一句话结论

修复 mamba_cache_mode=all 的 CPU offload 边界对齐，精确边界重算最后一个 token，避免输出变化。

## 动机

PR 正文标注 Fixes #51094。Root cause：`resolve_mamba_align_size()` 只对 `mamba_cache_mode="align"` 启用边界对齐；在 `"all"` 模式下，精确 chunk-boundary 的 lookup 可能返回完整 boundary 的同时恢复该 boundary 的 point state，Mamba 于是从一个已经消费了最后一个 prompt token 的状态继续，静默改变生成输出。

## 优化点

正文 Summary 列出三点：把 `mamba_cache_mode="all"` 与 `"align"` 一样纳入 OffloadingConnector hit window 约束；在精确 offload-chunk boundary 重算最后一个 token，而不是恢复已包含它的 Mamba point state；扩展既有 Mamba CPU-offload boundary 测试以覆盖两种 cache mode。结合 diff，`resolve_mamba_align_size()` 的判断由 `== "align"` 改为 `in ("align", "all")`，docstring 同步说明 "align" 或 "all" 时 hit window 需向下取整到 offloaded chunk size 的倍数；测试改名为 `test_mamba_cpu_offload_boundary` 并参数化 `mamba_cache_mode=["align", "all"]`，另设 `mamba_cache_dtype="float32"`、`mamba_ssm_cache_dtype="float32"` 做无损状态存储，使精确相等回归能把边界选择与低精度 Mamba checkpointing 隔离开。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: resolve_mamba_align_size() 条件由 == "align" 改为 in ("align", "all")，docstring 同步
tests/v1/kv_connector/unit/test_offloading_connector.py: test_mamba_align_cpu_offload 改名 test_mamba_cpu_offload_boundary 并参数化 align/all，设 float32 的 mamba cache dtype 做无损状态存储
```

（本 PR diff 仅涉及以上 2 个文件。）

```text
-        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
+        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in (
+            "align",
+            "all",
+        ):
```

## 性能声明

PR 正文未给出定量性能数据。正文 Testing 为正确性与回归验证，无 benchmark 数字，pack 的 perf claims regex 未检测到；测试条件见验证部分（H100、Nemotron-Nano-9B-v2、`mamba_cache_mode=all`、2816-token 精确边界）。

## 验证与回归风险

PR 引入/依赖的测试（正文 Testing）：所有适用于改动文件的 pre-commit hooks 通过（ruff、format、mypy、SPDX、forbidden imports、repository validation）；pytest collection 确认既有 GPU 测试扩展为 align 与 all 两个 case；直接 helper 验证 `none -> None`、`align -> 32`、`all -> 32`；H100 模型测试（当前 main `43c4bdca`、Nemotron-Nano-9B-v2、OffloadingConnector、`mamba_cache_mode=all`、2816-token exact boundary）中 cold run 与 CPU-refetch run 产出相同的 32 个 token ID，connector debug 显示 refetch 被约束到 1408 tokens；v0.25.1 上的 issue 复现作为 negative control（无此改动时精确边界 refetch 会改变输出）。PR 未注明回归风险；改动集中在 `mamba_cache_mode=all` 的 offload boundary 选择，可能受影响路径为 Mamba 混合 KV cache 的 CPU offload 精确边界命中与生成输出正确性；其他模型/硬件/精度正文未说明。

## 关联

正文标注 Fixes #51094；Duplicate check 提及 PR #50409（涉及 align-mode local replay-boundary caching，正文明确其不覆盖 all-mode CPU-offload hit-window 对齐），并说明目前没有 PR 修复 #51094。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
