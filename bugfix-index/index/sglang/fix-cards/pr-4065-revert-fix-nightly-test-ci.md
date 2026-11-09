# PR #4065: Revert "Fix nightly-test CI"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4065](https://github.com/sgl-project/sglang/pull/4065) |
| 作者 | merrymercy |
| 合并日期 | 2025-03-04 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论

整体回滚 #3826:删除 flashinfer backend 里给 KV buffer 做 dtype 转换的 `_to_dtype` 包装,恢复直接传 `get_kv_buffer` 结果。

## 问题现象

正文未描述具体现象。正文仅一句 "Reverts sgl-project/sglang#3826"(被回滚 PR 标题为 "Fix nightly-test CI"),未填写复现条件、影响面或原因。

## 根因

正文未披露根因。(由 diff 推断)被回滚的 #3826 曾在 `flashinfer_backend.py` 的 `forward_extend` 与 `forward_decode` 中,把 `token_to_kv_pool.get_kv_buffer(layer.layer_id)` 的返回先用新增的 `_to_dtype(..., q.dtype)` 转成与 query 相同的 dtype 再交给 flashinfer wrapper(针对 nightly 测试中 KV cache dtype 与 q dtype 不一致的场景);该转换在常规路径上引入了额外的显存拷贝/开销,本次回滚整体撤销,恢复原有直接传 KV buffer 的行为。

## 修复方案

按 revert 提交恢复 `python/sglang/srt/layers/attention/flashinfer_backend.py` 三处:

`forward_extend`(prefill_wrapper_paged 分支)与 `forward_decode`(decode_wrapper.forward)中,KV 参数从 `self._to_dtype(forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id), q.dtype)` 恢复为 `forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)`:

```diff
-                    self._to_dtype(
-                        forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
-                        q.dtype,
-                    ),
+                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
```

并删除整个 `_to_dtype` 辅助方法(`if kv_tuple[0].dtype != dtype: return tuple(t.to(dtype) for t in kv_tuple)` 的实现)。

## 影响与回归风险

- 影响窗口:含 #3826 改动的版本受影响(依据正文 revert 声明;正文未给出具体版本号)。
- 回归面:改动触及 flashinfer attention 后端的 extend 与 decode 两条前向路径。恢复后 KV buffer 不再被强制转换 dtype,nightly 测试中 #3826 想解决的问题(若为 dtype 不匹配导致的 CI 失败)可能复现;当 KV cache dtype 与 q dtype 确实不一致时,调用方语义回到 #3826 之前的原样。

## 关联

- [#3826](https://github.com/sgl-project/sglang/pull/3826)(正文声明被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
