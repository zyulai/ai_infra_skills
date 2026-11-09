# PR #28768: [BugFix] Fix PP performance and PP kv connector output regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28768](https://github.com/vllm-project/vllm/pull/28768) |
| 作者 | njhill |
| 合并日期 | 2025-11-17 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 5 |
| 分类 | KV Cache / Memory |
| bug 类型 | regression |

## 一句话结论
#26866 引入 PP 性能回归与 KV connector 输出丢失,重构执行/采样时序并透传 kv_connector_output 修复。

## 问题现象
正文披露两个由 #26866 引入的 pipeline-parallel 相关问题:1) 性能回归(issue #28270 报告);2) KV connector 输出无法从 worker 正确传播/聚合。正文未给出具体性能数字或报错日志。

## 根因
正文披露问题由 #26866 引入,但未详述两个问题的具体机制。由 diff 可推断:
- KV connector 输出问题:此前 `kv_connector_output` 被打包进 `ExecuteModelState`,仅在 `sample_tokens()` 中返回;当 `execute_model_state` 为 None(如 PP 非末 rank、无需采样的步)时该输出被直接丢弃,导致下游无法收到。
- 性能问题:Ray executor 在 `execute_model()` 阶段不触发 DAG 执行、只缓存 `scheduler_output` 等到 `sample_tokens()`,改变了原有的执行时序;engine core 侧 #28329 加的细粒度 tracing 上下文也让代码难以维护(正文称第一个 commit 直接 revert 这些 tracing 包裹)。

## 修复方案
- `vllm/v1/engine/core.py`:移除 `record_function_or_nullcontext` 细粒度 tracing 包裹;`step_with_batch_queue` 引入 `ec_producer` 判断,无采样需求的步直接把 `exec_future` 入队,并对不需结果的 future 挂 `_log_err_callback` 记录错误。
- `vllm/v1/executor/ray_executor.py`:抽出 `_execute_dag()`;`ec_producer` 或无调度 token 时立即执行 DAG,否则延迟到 `sample_tokens()`。
- `vllm/v1/worker/gpu_model_runner.py`:`kv_connector_output` 从 `ExecuteModelState` 移出为独立字段 `self.kv_connector_output`;`sample_tokens()` 在 `execute_model_state` 为 None 时仍透传该输出(PP with kv transfer 场景),空则返回 `EMPTY_MODEL_RUNNER_OUTPUT` 或其拷贝。
- `vllm/v1/worker/gpu_worker.py`:删除 `execute_model` 中的 kv_connector_output 透传逻辑(职责移入 model runner)。

## 影响与回归风险
- 影响窗口:正文未说明(问题由 #26866 引入)。
- 回归面:V1 PP 执行路径、KV connector 输出传播、Ray executor 异步 DAG 执行时序、`step_with_batch_queue` 异步调度路径、EC(ec_transfer_config)producer 场景;`sample_tokens()` 返回值分支增多,需关注非 PP 场景不受影响。

## 关联
- Fixes [#28270](https://github.com/vllm-project/vllm/issues/28270)
- 引入问题的 [#26866](https://github.com/vllm-project/vllm/pull/26866)
- 第一个 commit revert 的 [#28329](https://github.com/vllm-project/vllm/pull/28329)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
