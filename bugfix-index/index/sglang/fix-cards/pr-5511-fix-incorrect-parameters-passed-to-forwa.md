# PR #5511: Fix: Incorrect parameters passed to forward_batch_generation (#5506)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5511](https://github.com/sgl-project/sglang/pull/5511) |
| 作者 | u4lr451 |
| 合并日期 | 2025-04-18 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
EAGLE worker idle 分支给 `forward_batch_generation` 传参错误;改为直接传 `model_worker_batch` 并对齐返回值个数。

## 问题现象
正文未描述具体现象,仅说明动机为修复 "Incorrect parameters passed to TpModelWorker.forward_batch_generation"(关联 issue #5506)。按改动位置推断(由 diff 推断):问题出现在 EAGLE speculative decoding 下 batch 处于 idle forward 模式(如 DP attention/空闲分片场景)的路径,`eagle_worker.forward_batch_speculative_generation` 调用 target worker 时传参与函数签名不匹配。

## 根因
正文披露的根因:传给 `TpModelWorker.forward_batch_generation` 的参数不正确(对应 issue #5506),未给出更细节的分析。

具体表现(由 diff 推断):该函数签名已改为接收 `ModelWorkerBatch` 并只返回两个值 `(logits_output, next_token_ids)`,但 `eagle_worker.py` 的 idle 分支仍在用旧调用方式——先构造 `ForwardBatch.init_new(model_worker_batch, ...)` 再传入,并按三个返回值解包 `logits_output, next_token_ids, _`,函数末尾还多返回一个 `False`,与新签名不一致。

## 修复方案
改动在 `python/sglang/srt/speculative/eagle_worker.py` 的 `forward_batch_speculative_generation` idle 分支:

```diff
-            logits_output, next_token_ids, _ = (
-                self.target_worker.forward_batch_generation(
-                    ForwardBatch.init_new(
-                        model_worker_batch, self.target_worker.model_runner
-                    )
-                )
+            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
+                model_worker_batch
             )
-            return logits_output, next_token_ids, model_worker_batch.bid, 0, False
+
+            return logits_output, next_token_ids, model_worker_batch.bid, 0
```

即:直接传 `model_worker_batch`,按两个返回值解包,并把返回元组从 5 项改回 4 项。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围;按 diff 推断为 `forward_batch_generation` 签名调整(相关 #5506 改动)之后、本修复之前的版本。
- 回归面:仅影响 EAGLE worker 的 idle forward 路径;正常 draft/target 推理分支走 `forward_target_extend`,不受影响。若返回元组下游消费者仍期望 5 项,需要一并核对(本次改动将其对齐为 4 项)。

## 关联
- [#5506](https://github.com/sgl-project/sglang/issues/5506)(正文提到的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
