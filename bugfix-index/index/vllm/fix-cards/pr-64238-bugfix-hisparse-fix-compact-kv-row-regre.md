# PR #64238: [Bugfix][HiSparse] Fix compact KV row regression tests

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64238](https://github.com/vllm-project/vllm/pull/64238) |
| 作者 | khluu |
| 合并日期 | 2026-10-28 |
| labels | bug, hisparse, ready |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#58305 新增的三个 HiSparse 测试在 nightly 必挂(引用已删 helper、dtype 不匹配、self=None);仅修测试代码。

## 问题现象
Nightly main 构建 #97886 至 #98096 连续在 H100 与 B200 的 V1 Attention shard 上失败三个测试:
- `test_hisparse_index_group_preserves_row_width`:对已被移除的 `_create_side_stream` 报 `AttributeError`;
- `test_hisparse_gather_prefill_cache_prefers_resident_rows`:runtime 期望 `torch.float32` 却收到参数化的 `torch.uint8` cache;
- `test_hisparse_staged_prefill_cache_backs_min_attention_rows`:gather 实现改用 runtime cache ops 后,`self=None` 调到 `self.cache_ops`。
正文说明这些失败是确定性的测试缺陷,与后续无关的 main 提交无关。

## 根因
正文披露:失败是 #58305 合入的测试自身缺陷——#58305 合并前平台后端重构已删除 `_create_side_stream` / `_create_event` helper,测试仍在 monkeypatch 它们;gather-prefill 测试用默认 `float32` runtime 去接参数化的 FP8 `uint8` cache;第三个测试在 gather 路径改用 runtime cache ops 后仍以 `self=None` 直接调实例方法。属测试代码问题,非产品缺陷。

## 修复方案
仅改 `tests/v1/attention/test_sparse_mla_backends.py`:
- 删除对 `index_group_module._create_side_stream` / `_create_event` 的两行 monkeypatch;
- `test_hisparse_gather_prefill_cache_prefers_resident_rows` 构造 runtime 时传入参数化的 `kv_dtype`,使 FP8 `uint8` cache 绑定 `uint8` runtime:
```diff
-    runtime = _make_hisparse_runtime(row_width=row_width, block_size=block_size)
+    runtime = _make_hisparse_runtime(
+        row_width=row_width, block_size=block_size, kv_dtype=kv_dtype
+    )
```
- `test_hisparse_staged_prefill_cache_backs_min_attention_rows` 改为构建真实初始化的 runtime、`bind_source_cache(host_cache)` 后再调 `runtime.gather_prefill_cache(host_cache, plan)`,不再 `HiSparseRuntime.gather_prefill_cache(None, ...)`。

## 影响与回归风险
- 影响窗口:nightly #97886 起(缺陷由 #58305 引入)。
- 回归面:纯测试代码改动,`vllm/` 运行时代码零改动,对用户无影响;验证含 file 级 pre-commit 与更新后的 H100/B200 V1 Attention CI(首次构建 #97912 还暴露了第三个失效调用)。

## 关联
- 缺陷来源 [#58305](https://github.com/vllm-project/vllm/pull/58305)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
