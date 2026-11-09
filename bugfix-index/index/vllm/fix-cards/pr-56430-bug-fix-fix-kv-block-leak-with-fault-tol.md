# PR #56430: [bug-fix] Fix KV block leak with fault tolerance enabled in P/D disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56430](https://github.com/vllm-project/vllm/pull/56430) |
| 作者 | tiankaijin |
| 合并日期 | 2026-10-22 |
| labels | bug, ci/build, kv-connector, mrv2, ready, scheduler |
| 热度 | 👍 0 · 💬 18 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
P/D 分离部署开启 FT 后,worker 失败会丢 KV 传输进度并在响应队列留残渣,导致请求挂起与 KV block 泄漏;通过失败时抢救 KV 输出、drain-then-raise、延迟重放三类机制修复。

## 问题现象
将 FT 框架(#44428)扩展到 P/D 分离部署时发现:开启 KV transfer 后 worker 失败引发两个问题——① KV 传输进度丢失:`_model_forward` 抛异常时异常路径直接丢弃 `kv_connector_output`,在途传输的 `finished_sending`/`finished_recving`/`invalid_block_ids` 全部丢失;retry 恢复后调度器的 KV 记账与 connector 实际状态不再一致,请求可能挂起或泄漏 KV block。② 响应队列残留:executor 的 `get_response` 遇到第一个 `FAILURE` 立即 raise,不 dequeue 其余 worker 的响应;MQ 里遗留的陈旧响应会被下一次 RPC 消费,污染后续步骤。

## 根因
正文完整披露:异常处理路径把「worker 失败」当成整体丢弃,没有把失败 worker 已产生的 KV connector 输出抢救出来;同时 `get_response` 的 fail-fast 语义破坏了集体 RPC「所有 worker 响应必须成对消费」的不变量。修复的关键设计(均来自正文):失败时经新增的 `ExceptionWithKVConnectorOutput` / `ResponseStatus.FAILURE_WITH_KV_OUTPUT` 携带 KV 输出并暂存进 `KVOutputAggregator`;`get_response` 一律先 dequeue 所有 worker 响应再 raise(drain-then-raise,覆盖所有集体 RPC);暂存输出在下一个成功 step 经正常 `aggregate()` 循环重放(defer-and-replay);恢复时 `EngineCoreSentinel` 逐个消费 batch queue 中 future 的 KV 输出(consume, don't drop)而非整体 clear。被打捞的 `kv_cache_events` 直接丢弃:丢事件只损失一次 cache 命中,重放它们反而可能扰动存活 worker 的事件 quorum(如 LMCache `count == num_workers` 精确匹配)。

## 修复方案
主要改动:
- `vllm/v1/outputs.py`:`AsyncModelRunnerOutput` 新增抽象方法 `get_kv_connector_output()`;`vllm/v1/worker/gpu/async_utils.py`、`vllm/v1/worker/gpu_model_runner.py` 各实现补齐。
- `vllm/v1/executor/multiproc_executor.py`:新增 `ExceptionWithKVConnectorOutput` 与 `ResponseStatus.FAILURE_WITH_KV_OUTPUT`;`WorkerProc.enqueue_output` 在 `get_output()` 抛异常时提取 KV 输出;`get_response` 改为 drain-then-raise,失败时收集 KV 输出交 `ft_defer_kv_connector_outputs` 暂存后再 raise(FT 关闭或 executor 已 failed 时保留原 raise-immediately 行为)。
- `vllm/distributed/kv_transfer/kv_connector/utils.py`:`KVOutputAggregator` 新增 `_salvaged_outputs` 与 `ft_defer_kv_connector_outputs()`,暂存输出在下次 `aggregate()` 中与实时输出一起参与聚合,`aggregate()` 本身不变。
- `vllm/v1/fault_tolerance/engine_core_sentinel.py`:恢复时 `_drain_batch_queue()` 逐个消费 future 的 `kv_connector_output`,经 `scheduler.update_from_output(SchedulerOutput.make_empty(), ModelRunnerOutput.with_kv_conn_output_only(...))` 只应用 KV 传输状态。
- 新增/扩展单测:`tests/v1/kv_connector/unit/test_output_aggregator.py`、`tests/v1/executor/test_executor.py`、`tests/v1/fault_tolerance/test_engine_core_sentinel.py`(新建);`.buildkite/test_areas/fault_tolerance.yaml` CI 接线。

```diff
+    def ft_defer_kv_connector_outputs(
+        self, kv_connector_outputs: list[KVConnectorOutput | None]
+    ) -> None:
+        """Stash KV connector outputs from failed workers for replay at the
+        next aggregate().
+        """
+        for o in kv_connector_outputs:
+            if o is not None:
+                o.kv_cache_events = None
+        self._salvaged_outputs.extend(o for o in kv_connector_outputs if o)
```

## 影响与回归风险
- 影响窗口:正文指出问题随 FT 框架(#44428)用于 KV transfer 的 P/D 分离部署出现;具体版本号未提及。
- 回归面:executor 的 `get_response` 失败语义(所有集体 RPC)与 `KVOutputAggregator` 聚合逻辑,影响所有使用 KV connector 的 v1 部署;FT 关闭或 executor 已 failed 时行为保持原样(正文明示);`kv_cache_events` 在失败 step 被丢弃是有意行为(代价仅为 cache miss);CI 侧 fault_tolerance 流水线新增 sentinel 测试与 `vllm/distributed/kv_transfer/`、`vllm/v1/executor/` 源文件依赖。人工验证:Qwen3-30B-A3B,1 prefill + 1 decode,MooncakeConnector + FT,step 250/5000 注入 RuntimeError 后查询 `/fault_tolerance/status` 并 retry 恢复。

## 关联
- FT 框架: https://github.com/vllm-project/vllm/pull/44428

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
