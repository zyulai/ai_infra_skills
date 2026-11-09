# PR #8050: [Core] Optimize Async + Multi-step

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2024-09-03 |
| 分类 | Model-specific Fast Path |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/8050 |

## 一句话结论

把 async output processing 覆盖到 multi-step 的全部步骤(含最后一步与 prompt 执行),Llama3-8B H100 TPOT 44.48→32.38 ms(-37%)。

## 动机

multi-step decoding(一次调度连续执行多个 decode step,服务于 speculative decoding 的 lookahead)与 async output processor 组合时,异步只覆盖 multi-step 的"前序" decode 步骤:生成最终结果的最后一步与 prompt 执行仍走同步后处理,CPU 后处理与 GPU 前向无法重叠,收益此前不足 10%。

## 优化点

统一异步路径:删除 `use_async_and_multi_step` 标志与独立的 `async_callback_multi_step`,所有步骤共用 `async_callbacks`;`SchedulerContext.output_queue` 元组扩为 `(outputs, metadata, scheduler_outputs, is_async, is_last_step)`,multi-step worker 的 `_async_process_outputs`/`_final_process_outputs` 在每个 output pythonize 完成后把它追加进 `ctx.output_queue` 并立刻触发 callback,形成 callback→pythonize 交替链以最大化与 GPU 的重叠(`multi_step.py` 的 `process_outputs` 相应支持 `is_async`,只做 detokenize + stop check)。引擎侧:`AsyncLLMEngine` 注册 `process_request_outputs_callback`,让完成的 request output 在 `_process_model_outputs` 内直接 append 到 asyncio 队列;finished 序列的 RequestOutput 只在 `finished_now` 时创建并即时释放 seq group。pythonization 引入 `PythonizationCache`(PyObjectCache)复用 `SequenceOutput`/`CompletionSequenceGroupOutput` 对象,减少逐步骤的 Python 分配。

## 关键改动

```text
vllm/engine/llm_engine.py: output_queue 扩展 + _process_model_outputs 重构(ctx 参数化、finished_now)
vllm/engine/async_llm_engine.py: process_request_outputs_callback 快速出队
vllm/worker/multi_step_model_runner.py: callback/pythonize 交替链 + PythonizationCache
vllm/engine/output_processor/multi_step.py: process_outputs 支持 is_async 分支
```

## 性能声明

Llama3 8B、H100、ShareGPT 数据集、500 requests:multi-step(无 async)→ multi-step + async:output token throughput 5334.15 → 6261.00 tok/s,request throughput 26.57 → 31.18 req/s,mean TPOT 44.48 → 32.38 ms(PR 称 37% 改善,整体约 ~28% vs main),mean ITL 272.29 → 217.58 ms,TTFT 基本持平。

## 验证与回归风险

PR 正文 TODO 自述"Cleanup the PR / Verify all tests pass"尚未完成,无新增测试。风险面:改动集中在 v0 引擎核心调度/后处理路径(`llm_engine.py`、`async_llm_engine.py`),影响所有 async 输出场景;multi-step 路径显式 assert 不支持 logits processors、不支持 beam search;`ExecuteModelRequest`/`ModelInputForGPU` 字段变更涉及 worker 间序列化。

## 关联

同区域:multi-step decoding(为 speculative decoding 提供 lookahead slots)与 async output processing 线;后续 v1 引擎对 async scheduling 的重写取代此路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
