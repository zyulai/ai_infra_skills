# PR #51100: [Bugfix] Fix Mamba all-mode CPU offload boundary alignment

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51100](https://github.com/vllm-project/vllm/pull/51100) |
| 作者 | jairitAge |
| 合并日期 | 2026-08-07 |
| labels | bug, kv-connector, ready |
| 热度 | 👍 2 · 💬 33 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
`mamba_cache_mode="all"` 未纳入 OffloadingConnector 命中窗口对齐,精确块边界取回的 point state 已含末 token,输出被静默改变;补上该模式修复。

## 问题现象
使用 OffloadingConnector + `mamba_cache_mode="all"` 的 Mamba 模型,当 prompt 长度恰好落在 offload chunk 边界(如测试中的 2816-token 精确边界)时,CPU 回取(refetch)会命中完整边界并恢复该边界处的 point state,生成的输出被静默改变(不报错)。正文引用 issue #51094 在 v0.25.1 上的复现作为阴性对照:不修此改动,精确边界的 CPU refetch 会改变输出。

## 根因
正文披露:`resolve_mamba_align_size()` 只对 `mamba_cache_mode="align"` 启用边界对齐(hit window 向下取整到 offload chunk 的倍数);在 `"all"` 模式下,精确 chunk 边界的查找会返回完整边界、同时恢复该边界处的 point state,而 Mamba 从一个已经消费了最后一个 prompt token 的状态续算,静默改变生成输出。缓存的 Mamba 条目是 point state,恢复后需要重算的末 token 已包含其中,对其他 attention 类型无此问题(其 hit block 内含全部 token KV)。

## 修复方案
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` 的 `resolve_mamba_align_size`:把 `"all"` 与 `"align"` 同等对待,两种模式的 hit window 都向下取整到 offloaded chunk 大小的倍数:

```diff
-        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
+        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in (
+            "align",
+            "all",
+        ):
```

效果为在精确 offload-chunk 边界处重算最后一个 token,而不是恢复已包含它的 Mamba point state。同时把既有测试扩展为 `test_mamba_cpu_offload_boundary` 并参数化覆盖 `align`/`all` 两种模式,用 float32 无损 state 存储隔离低精度 checkpoint 干扰。

## 影响与回归风险
- 影响窗口:正文提到 issue 在 v0.25.1 复现(作为阴性对照),未给出其他受影响版本范围。
- 回归面:仅影响使用 OffloadingConnector 且 MambaSpec `mamba_cache_mode` 为 `all` 的请求命中窗口计算;`align` 模式行为不变,非 Mamba 后端不受影响。验证:H100 + Nemotron-Nano-9B-v2(main `43c4bdca`)2816-token 精确边界下冷启动与 CPU-refetch 输出 32 个 token ID 一致,refetch 被约束到 1408 tokens。

## 关联
- Fixes https://github.com/vllm-project/vllm/issues/51094
- 相关(非修复对象)https://github.com/vllm-project/vllm/pull/50409

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
