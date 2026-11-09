# PR #15227: [IDLE FORWARD][Indexer] Fix forward_idle bs mismatch issue in DeepseekV3.2's NSAIndexer

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15227](https://github.com/sgl-project/sglang/pull/15227) |
| 作者 | YAMY1234 |
| 合并日期 | 2026-01-10 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

DP Attention IDLE 前向复用陈旧 attention 元数据致 NSA Indexer 断言失败,补 init_forward_metadata 修复。

## 问题现象

在 DP Attention 场景下,当部分 DP rank 没有实际请求时进入 IDLE 模式。经 `prepare_mlp_sync_batch` 处理后,这些 IDLE batch 会被填充(batch_size > 0)以参与 MLP sync。此时 `forward_idle` 没有重新初始化 attention metadata,继续沿用上一批(batch_size 不同)的陈旧 metadata,导致 `deep_gemm.fp8_paged_mqa_logits` 等 attention kernel 内断言失败。触发条件是 DeepseekV3.2 的 NSAIndexer 配合 DP Attention IDLE 模式。错误示例:

```
RuntimeError: Assertion error (attention.hpp:160): batch_size == batch_size_ and batch_size == batch_size__
```

## 根因

正文披露:`forward_idle` 未重新初始化 attention metadata,导致使用与当前 padded IDLE batch 不一致(不同 batch_size)的上一批陈旧 metadata。

## 修复方案

在 `python/sglang/srt/model_executor/model_runner.py` 的 `forward_idle` 中,当 `forward_batch.batch_size > 0` 时调用 `self.attn_backend.init_forward_metadata(forward_batch)`,保证 metadata 与当前 padded IDLE batch 一致。

关键 diff 摘录:

```diff
     def forward_idle(
         self, forward_batch: ForwardBatch, pp_proxy_tensors=None
     ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
+        # In DP Attention, IDLE batches are padded (batch_size > 0) for MLP sync.
+        # in this case, we need to reinit the forward metadata, otherwise the stale
+        # metadata causes batch_size mismatch in attention kernel(e.g. NSA Indexer).
+        if forward_batch.batch_size > 0:
+            self.attn_backend.init_forward_metadata(forward_batch)
+
         kwargs = {}
```

作者用 gsm8k 基准验证(`--num-shots 20 --num-questions 1319 --parallel 1319`),开启与关闭 `--enable-cuda-graph` 各跑 5 次问题均未复现,Accuracy 0.957、Invalid 0.000。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:所有走 `forward_idle` 且 batch_size > 0 的路径(DP Attention 下 padded IDLE batch,含开启 PP 传 `pp_proxy_tensors` 的场景)现在都会在每次 idle 前向重建 attention metadata,可能带来少量额外开销;行为上使 metadata 与当前 batch 一致,影响 NSA Indexer 等 attention kernel 的输入校验。

## 关联

- Issue #15132: https://github.com/sgl-project/sglang/issues/15132(正文 Motivation 提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
