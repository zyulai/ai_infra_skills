# PR #10985: Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10985](https://github.com/sgl-project/sglang/pull/10985) |
| 作者 | yhyang201 |
| 合并日期 | 2025-10-01 |
| labels | high priority, run-ci |
| 热度 | 👍 5 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | startup |

## 一句话结论
Qwen3-VL 启动即失败:MRotaryEmbedding 不接受 fused_set_kv_buffer_arg;按 rotary 类型关闭 fused kv 写入路径。

## 问题现象
Qwen3-VL 启动(launch)即失败,模型无法正常拉起。触发条件:模型复用 Qwen3-MOE 的实现接口,但其 rotary embedding 实际是 `MRotaryEmbedding` 而非 `RotaryEmbedding`。起因是 #10749 在 Qwen3-MOE 中把 `fused_set_kv_buffer_arg` 传给 `q, k = self.rotary_emb` 调用,而该参数只有 `RotaryEmbedding` 接受,于是 Qwen3-VL 一启动就报错(正文表述为 "caused the model to fail to launch")。

## 根因
正文披露:#10749 给 Qwen3-MOE 的 rotary_emb 调用传入 `fused_set_kv_buffer_arg`,但只有 `RotaryEmbedding` 支持该参数;Qwen3-VL 虽调用 Qwen3-MOE 接口,实际使用 `MRotaryEmbedding`,不接受该参数,导致模型无法启动。

## 修复方案
两处改动(按 diff):
1. `python/sglang/srt/layers/rotary_embedding.py`:`MRotaryEmbedding.forward` 增加 `fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None` 形参,非 None 时直接断言报错 `"save kv cache is not supported for MRotaryEmbedding."`,防止再次误传。
2. `python/sglang/srt/models/qwen3_moe.py`:`Qwen3MOEAttention.__init__` 中按 `isinstance(self.rotary_emb, MRotaryEmbedding)` 设置 `self.compatible_with_fused_kv_buffer`;`forward_prepare` 与 `forward_core` 把 fused set kv buffer 的启用条件改为 `enable_fused_set_kv_buffer(forward_batch) and self.compatible_with_fused_kv_buffer`,不兼容时 `save_kv_cache=True` 走常规写 KV 路径。

```diff
-            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
+            save_kv_cache=not (
+                enable_fused_set_kv_buffer(forward_batch)
+                and self.compatible_with_fused_kv_buffer
+            ),
```

## 影响与回归风险
- 影响窗口:正文未给出版本范围,即 #10749 引入 fused_set_kv_buffer_arg 之后、本修复之前。
- 回归面:按 diff,影响使用 `MRotaryEmbedding` 的 Qwen3 系模型(如 Qwen3-VL)的 fused kv buffer 快路径——现回落到 `save_kv_cache=True` 的非融合路径;`RotaryEmbedding` 路径行为不变;`MRotaryEmbedding.forward` 新增断言,任何向其传 `fused_set_kv_buffer_arg` 的调用会立即报错。

## 关联
- [#10749](https://github.com/sgl-project/sglang/pull/10749)(引入问题的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
