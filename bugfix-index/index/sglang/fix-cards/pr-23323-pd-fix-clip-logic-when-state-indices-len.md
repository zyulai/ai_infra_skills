# PR #23323: [PD] Fix clip logic when state indices lens are mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23323](https://github.com/sgl-project/sglang/pull/23323) |
| 作者 | ShangmingCai |
| 合并日期 | 2026-04-21 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Mooncake PD 传输中 SWA/NSA 索引长度不一致时 clip 逻辑失效,`group_concurrent_contiguous` 抛错致传输失败;双向 clip 修复。

## 问题现象
SWA/NSA hybrid 模型在 PD disaggregation 下 KV 传输失败(Fix #18727)。触发条件:`prefill_state_indices` 与 `req.dst_state_indices` 长度不同时,下游 `_send_kvcache_generic` 路径调用 `group_concurrent_contiguous`,其内部

```python
brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
```

对两个不同 shape 的数组做 bitwise-OR 会抛异常,导致传输失败。

## 根因
正文披露:`MooncakeKVManager.maybe_send_extra` 中针对 SWA/NSA hybrid 模型的 clip 逻辑只处理了长度失配的一个方向,且该分支实际是 no-op。原代码:

```python
if len(prefill_state_indices) < len(req.dst_state_indices):
    logger.warning(...)
    prefill_state_indices = prefill_state_indices[: len(req.dst_state_indices)]
```

两个问题:1) `prefill < dst` 时,用大于自身长度的上界切片是 Python no-op,列表原样返回,两侧仍然不匹配;2) 相反方向(`prefill > dst`)完全没有处理。

## 修复方案
`python/sglang/srt/disaggregation/mooncake/conn.py` 的 `maybe_send_extra`(正文 Modifications 与 diff 一致):
- 两个方向都 clip,使传入 `_send_kvcache_generic` 前两侧索引一定对齐:`len(prefill) > len(dst)` 时把 `prefill_state_indices` 截到 `len(dst)`;`len(prefill) < len(dst)` 时把 `dst_state_indices` 截到 `len(prefill)`。
- 引入局部变量 `dst_state_indices` 取代原地修改 `req.dst_state_indices`,避免 clip 作为副作用改动 `TransferInfo`。
- 正文说明:Mamba / MLA 分支行为不变,只动 SWA/NSA hybrid 路径。

```diff
-            if len(prefill_state_indices) < len(req.dst_state_indices):
+            dst_state_indices = req.dst_state_indices
+            if len(prefill_state_indices) > len(dst_state_indices):
...
-                prefill_state_indices = prefill_state_indices[
-                    : len(req.dst_state_indices)
-                ]
+                prefill_state_indices = prefill_state_indices[: len(dst_state_indices)]
+            elif len(prefill_state_indices) < len(dst_state_indices):
...
+                dst_state_indices = dst_state_indices[: len(prefill_state_indices)]
```

## 影响与回归风险
- 回归面:仅 `maybe_send_extra` 的 SWA/NSA hybrid 分支;索引被截断意味着超长一侧的尾部索引不再发送(语义上与两侧对齐的需求一致,由 diff 推断);`req.dst_state_indices` 不再被就地修改,依赖原对象被修改这一副作用的行为会受影响(由 diff 推断)。

## 关联
- [#18727](https://github.com/sgl-project/sglang/issues/18727)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
