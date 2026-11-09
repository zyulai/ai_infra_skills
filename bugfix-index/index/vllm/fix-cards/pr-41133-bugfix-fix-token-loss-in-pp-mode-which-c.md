# PR #41133: [Bugfix] Fix token loss in PP mode which causes degraded accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41133](https://github.com/vllm-project/vllm/pull/41133) |
| 作者 | starkwj |
| 合并日期 | 2026-05-06 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
PP 模式 `_update_states` 误用 `num_computed_tokens` 更新 `num_tokens_no_spec`,condense 时丢失剩余 prompt token 导致精度下降;改按 `num_tokens_no_spec` 追加修复。

## 问题现象
PP 模式在并发请求下精度下降:gsm8k(strict-match)Qwen3-8B、num_concurrent 256 时,TP 为 0.8741,PP 2 + no-async-scheduling 为 0.8757,PP 2(num_concurrent 2)为 0.8741,而 PP 2 默认配置仅 **0.8324**。触发条件:PP + 默认 async-scheduling(或长 prompt + 较小 `max-num-batched-tokens` 下的 no-async 模式),chunked prefill 与批内请求重排同时发生时。

## 根因
正文披露:`gpu_model_runner.py` 中 `num_tokens_no_spec` 应表示当前无 spec 的 token 数,但 PP 模式下 `_update_states` 错误地按 `num_computed_tokens` 更新它;chunked prefill 时 `num_computed_tokens` 只是已计算的 chunk 数,小于全部 prompt token 数(真实 `num_tokens_no_spec`)。同时 `token_ids_cpu` 的更新下标也不正确:async-scheduling 下 `new_token_ids` 为空、不写入;no-async 下用同一段 prompt chunk 覆盖,均无意义。正确逻辑在 `_bookkeeping_sync`(非 PP 与 PP 末级;chunked prefill 时该操作因无有效采样 token 被 `discard_request_mask` 过滤跳过)。随后 `_update_states` 调用 `condense()` 把尾部有效请求挪到前排空槽并复制 token,但 token 数按 `num_tokens_no_spec` 计算——于是只复制了 `num_computed_tokens` 个 token,剩余 prompt token 丢失(被空槽里的无效 token 顶替)。no-async 下因 unscheduled 请求会先 remove 再 add_request、占住前排空槽,swap 概率大减,所以分数看似正常,但长 prompt + 小 `max-num-batched-tokens` 时仍可能触发。

## 修复方案
`vllm/v1/worker/gpu_model_runner.py` `_update_states`(非末级 rank 分支):起点改用 `num_tokens_no_spec[req_index]`,终点取 `max(start, num_computed_tokens + len(new_token_ids))`,仅在终点前进时把 `new_token_ids` 尾部 token 追加到 `token_ids_cpu`,并置 `is_token_ids`、推进 `num_tokens_no_spec`,逻辑对齐 `_bookkeeping_sync`:

```diff
-                start_token_index = num_computed_tokens
-                end_token_index = num_computed_tokens + len(new_token_ids)
-                self.input_batch.token_ids_cpu[
-                    req_index, start_token_index:end_token_index
-                ] = new_token_ids
-                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
+                start_token_index = self.input_batch.num_tokens_no_spec[req_index]
+                end_token_index = max(
+                    start_token_index,
+                    num_computed_tokens + len(new_token_ids),
+                )
+                if end_token_index > start_token_index:
+                    if new_token_ids:
+                        ...append new_token_ids[-num_new_tokens:]...
+                    self.input_batch.is_token_ids[
+                        req_index, start_token_index:end_token_index
+                    ] = True
+                    self.input_batch.num_tokens_no_spec[req_index] = end_token_index
```

另在 `gpu_input_batch.py` `update_req_spec_token_ids` 中补写 `is_token_ids` 标记,并新增两个 PP rank 状态一致性测试。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅改 PP 非末级 rank 的 `_update_states` token 缓冲维护路径及 spec token 的 `is_token_ids` 标记;末级 rank 与非 PP 路径不变。测试结果:no-async-scheduling 分数不变(0.8810/0.8757),性能 bench 无变化;gsm8k flexible-extract 由 0.8370 升至 0.8855。

## 关联
- Maybe related issue: https://github.com/vllm-project/vllm/issues/28839
- Maybe related issue: https://github.com/vllm-project/vllm/issues/38903

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
