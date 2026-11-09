# PR #30125: [MLX] Fix FakeOverlapScheduler test stub broken by forward_ct accounting

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30125](https://github.com/sgl-project/sglang/pull/30125) |
| 作者 | LijuanTang94 |
| 合并日期 | 2026-07-06 |
| labels | apple-silicon |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论

#29217 让 `_finalize_mlx_pending_job` 记账 `forward_ct`,MLX 测试 stub 缺该属性致单测报错;补 stub 属性修复。

## 问题现象

MLX 单测 `test_attention_patching.py` 中 `TestMlxOverlapScheduler` 的 `test_finalize_pending_job_updates_scheduler_last_batch` 失败,报错:

```
AttributeError: 'FakeOverlapScheduler' object has no attribute 'forward_ct'
```

该测试对真实 stub 调用 `_finalize_mlx_pending_job` 时崩溃。问题是 #29217 合入时就潜伏的(当时还没有 macOS MLX CI lane 跑这套测试),直到 #29691 落地、`mlx-unit-test` lane 开始对触发它的 PR 变红才暴露。

## 根因

按 PR 正文:#29217 让 `_finalize_mlx_pending_job` 前进 `self.forward_ct` 并调用 `self.profiler_manager._profile_batch_predicate(...)`(对齐 `run_batch()` 的记账),但既有的 `FakeOverlapScheduler` stub 两个属性都没定义,调用即抛 `AttributeError`。属测试 stub 未跟上 mixin 新增的记账接口,生产代码无恙。

## 修复方案

改 `test/registered/unit/hardware_backend/mlx/test_attention_patching.py` 的 `FakeOverlapScheduler.__init__`:stub `forward_ct = 0` 和一个暴露 no-op `_profile_batch_predicate` 的 `profiler_manager`,沿用本文件既有的 `SimpleNamespace` 写法,与新 `test_scheduler_mixin.py` 里用 `MagicMock` 的接线一致:

```diff
             self.last_batch = None
             self.processed_batch = None
             self.processed_result = None
+            self.forward_ct = 0
+            self.profiler_manager = SimpleNamespace(
+                _profile_batch_predicate=lambda batch: None
+            )
```

纯测试修复,不改生产代码。

## 影响与回归风险

- 影响窗口:#29217 合入后潜伏,#29691 启用 macOS MLX CI lane 后显现。
- 回归面:仅 MLX 单测的 stub;生产调度器、profiler 行为不受任何影响。作者在 `upstream/main` 本地验证:`test_attention_patching.py` 由 1 errored 变为 39/39 通过,`TestMlxOverlapScheduler` 全绿。

## 关联

- #29217(引入 forward_ct 记账的 PR)
- #29691(启用 mlx-unit-test lane、使问题暴露的 PR)
- 新 stub 参照:test_scheduler_mixin.py(正文提及,未附链接编号)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
