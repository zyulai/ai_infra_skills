# PR #7021: [WA] fix output data is nan in CI test "test_moe_eval_accuracy_large.py"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7021](https://github.com/sgl-project/sglang/pull/7021) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-06-10 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
运行 `test_moe_eval_accuracy_large.py` 时输出变 NaN;aiter 后端把 `kv_indices` 的 padding 从 0 改为 pool 末索引以绕过 CUDA graph capture 下 0 位被置 NaN 的问题(Workaround)。

## 问题现象
正文只写了「Fix the NAN output in running test_moe_eval_accuracy_large.py」,未给出更详细的复现与日志:运行 CI 测试 `test_moe_eval_accuracy_large.py` 时,输出数据为 NaN。标题中的 [WA] 表明这是 workaround 性质的修复。

## 根因
正文 Modifications 一节为空,未披露根因。根因说明出现在 diff 内注释中(由 diff 推断,以下转述 diff 注释原文):`mha_batch_prefill` 会读取 128 个数据做计算,真实数据不够长时使用原本的 padding 值 0;但在 CUDA graph capture 模式下,0 的位置会被置成 NaN(noqa),从而导致输出张量的值变成 NaN。

## 修复方案
改 `python/sglang/srt/layers/attention/aiter_backend.py`(AMD aiter attention backend)两处:
1. `__init__` 中记录 KV pool 的最后一个索引:
```python
self.pool_size = (
    model_runner.token_to_kv_pool.size + model_runner.token_to_kv_pool.page_size
) - 1
```
2. `update_single_wrapper` 中 `kv_indices` 由 `torch.empty(paged_kernel_lens_sum + 256, ...)` 改为用 pool 末索引填充:
```diff
-            kv_indices = torch.empty(
-                paged_kernel_lens_sum + 256,
+            kv_indices = torch.full(
+                (paged_kernel_lens_sum + 128,),
+                self.pool_size,
                 dtype=torch.int32,
                 device=req_pool_indices.device,
             )
```
即 padding slot 不再是 0,而是 pool 的最后一个有效索引,使 mha_batch_prefill 多读的 128 个数据不会落到会被置 NaN 的 0 上。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:仅 aiter attention backend 的 `update_single_wrapper` 路径。padding 值从 0 变为 `pool_size`,padding 容量从 `+256` 变为 `+128`(由 diff 可见);下游若对 `kv_indices` 尾部槽位的取值有 0 的假设,行为会改变。正文未提供测试结果或验证说明(Modifications 为空,checklist 未附实际数据)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
