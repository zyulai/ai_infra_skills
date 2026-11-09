# PR #38726: [Bugfix][Core] Fix stuck chunked pipeline parallelism with async scheduling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38726](https://github.com/vllm-project/vllm/pull/38726) |
| 作者 | starkwj |
| 合并日期 | 2026-04-17 |
| labels | bug, ready, v1 |
| 热度 | 👍 4 · 💬 4 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
PP + async-scheduling 下 CPP 被 sampled tokens 广播卡死;对纯 chunked prefill 批跳过收发广播,长上下文 prefill 吞吐提升约 3.3 倍。

## 问题现象
开启 pipeline parallelism 并使用默认设置(`enable-chunked-prefill`、`async-scheduling`)时,当前实现无法真正启用 chunked pipeline parallelism(CPP):流程被 `sample_tokens` 中的广播操作卡住(stuck)。启用 `async-scheduling` 后,各 PP rank 会经由 `sampled_tokens` 广播从最后一个 PP rank 接收采样 token,这对 decode 合理,但对 chunked prefill 无意义——非最后一个 chunk 的 prefill 采样 token 会被丢弃;于是前面的(非最后)PP rank 要等最后 PP rank 完成,无法继续 prefill 下一个 chunk,流水线停滞。作者用 `vllm serve` GLM-4.7-W8A8(-pp 4, -tp 2)、`vllm bench serve` random-input-len 65536 / random-output-len 1 复测:修复前 input token throughput 1392.32 tos/s、TTFT 47068.74 ms;修复后 4578.14 tos/s、TTFT 14313.72 ms。

## 根因
正文披露:非最后一个 chunk 的 chunked prefill 并不需要 sampled token,但 `_pp_broadcast_prev_sampled_token_ids` / `_pp_receive_prev_sampled_token_ids_to_input_batch` 路径上无条件执行 `torch.distributed.broadcast`,让非末级 PP rank 被迫等待末级 rank 的采样结果,阻塞了下一 chunk 的 prefill。

## 修复方案
`vllm/v1/worker/gpu_model_runner.py`:新增 `_is_all_reqs_chunked_prefill()`,复用 `prepare_inputs` 中判定请求不应采样的 `discard_request_mask`:

```python
def _is_all_reqs_chunked_prefill(self) -> bool:
    num_reqs = self.input_batch.num_reqs
    return bool(self.discard_request_mask.np[:num_reqs].all())
```

在末级 rank 的广播和接收侧各加一个跳过条件,仅当"所有调度请求都是 chunked prefill"时不做广播/接收:

```diff
-        torch.distributed.broadcast(
-            sampled_token_ids, src=pp.rank, group=pp.device_group
-        )
+        # Skip for chunked prefill: sampled tokens are dummy
+        # and will be discarded, no need to broadcast.
+        if not self._is_all_reqs_chunked_prefill():
+            torch.distributed.broadcast(
+                sampled_token_ids, src=pp.rank, group=pp.device_group
+            )
```

接收侧 `torch.distributed.broadcast(recv, src=pp.last_rank, ...)` 同样加上该跳过。准确率经 lm_eval gsm8k(0.8832→0.8848 flexible-extract)与 longbench_single 验证无回退。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动仅涉及 PP + async-scheduling 下 sampled tokens 的广播/接收;当批次中混有需要采样的请求时行为不变,只有全部请求都是(非最后 chunk 的)chunked prefill 时才跳过。作者自述复用既有判定逻辑"应该安全(不 100% 确定)";longbench 指标存在本身波动。

## 关联
- Related issue: https://github.com/vllm-project/vllm/issues/34886
- Related link: https://github.com/vllm-project/vllm/pull/33356
- Related link: https://www.lmsys.org/blog/2026-01-15-chunked-pipeline

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
