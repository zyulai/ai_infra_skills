# PR #58975: [Bugfix][Pooling] Handle multimodal cache misses without crashing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58975](https://github.com/vllm-project/vllm/pull/58975) |
| 作者 | Lcy0801 |
| 合并日期 | 2026-10-02 |
| labels | bug, frontend, ready, mrv2 |
| 热度 | 👍 0 · 💬 14 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
Pooling 请求遇到多模态缓存 miss 时,OutputProcessor 误走 generation 路径触发 `detokenizer is not None` 断言,杀掉共享 output-handler 任务并关停服务;改为结构化 RequestError 并让批次继续。

## 问题现象
Fixes #33865:EngineCore 报告多模态 processor 缓存 miss 时,返回带 `mm_cache_miss_hashes` 的终态 `FinishReason.ERROR` 输出。Pooling 请求没有 detokenizer,但错误输出不含 pooling tensor 时 OutputProcessor 会跟随 generation 路径处理,触发 `req_state.detokenizer is not None` 断言失败。后果是断言杀掉共享的 output-handler 任务,进而关停整个 server——一个请求的多模态缓存 miss 可以打倒整个服务实例。

## 根因
正文披露:pooling 请求的终态错误输出(无 pooling tensor)被 OutputProcessor 按 generation 路径处理,而 pooling 请求状态里 detokenizer 为 None,断言失败;共享 output-handler 任务被杀导致服务器关闭。即 pooling 的终态失败路径没有被结构化处理,错误被当作引擎内部崩溃而不是请求级可重试失败。

## 修复方案
1. `vllm/outputs.py`:新增冻结 dataclass `RequestError`(code/message/retryable);`PoolingRequestOutput` 增加 `error` 字段,三个特化输出(Embedding/Classification/Scoring)的 `from_base` 在有 error 时用空占位值(`[]`/`0.0`)并透传 error,避免失败输出被再次解析。
2. `vllm/v1/engine/output_processor.py`:`process_outputs` 在 `detokenizer is None` 且输出带 `mm_cache_miss_hashes` 且 `finish_reason == ERROR` 时构造 `RequestError(code="multimodal_cache_miss", retryable=True)`,并把 pooling_output 置为 `EMPTY_CPU_TENSOR`,走正常 pooling 出口而非 generation 断言路径;`make_request_output`/`_new_request_output` 透传 error。
3. `vllm/entrypoints/pooling/base/serving.py` `_collect_batch`:结果带 error 时,可重试抛 `RetryableRequestError`、否则抛 `GenerationError`;`exceptions.py` 新增 `RetryableRequestError`(HTTP 503),`error_response.py` 将其映射为 `ServiceUnavailableError`,客户端可重试。
4. scoring 后处理短路:Bi-encoder/Late-interaction(`_pair_error_output`)与 Jina ranking 的 `_post_process` 在打分计算前检查 error 并原样透传失败输出,占位 tensor 不再进入余弦相似度/MaxSim 计算。
5. 缓存与清理:`v1/engine/llm_engine.py` 同步 `step()` 使 `mm_cache_miss_hashes` 对应的 processor 缓存失效(对齐 AsyncLLM 行为);`late_interaction_runner.py` 新增 `release_queries` 定向释放查询缓存(幂等),`flash_late_interaction` 失败时 `_cleanup_flash_late_interaction` 先 abort 文档再 collective_rpc 释放查询;文档 ID 改为请求级命名空间 `late-interaction-{uuid}-doc-{i}`,避免共享 `X-Request-Id` 的并发请求在清理时互相误杀。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:pooling 输出处理与 scoring 服务路径(含 bi-encoder、late-interaction、Jina ranking)、HTTP 错误码映射(可重试 pooling 失败从 500 变为 503)、同步 LLMEngine 缓存失效与 late-interaction 查询缓存释放。成功请求的输出与精度不变。作者说明因 macOS 无法装 CUDA 依赖未做 Qwen3-VL-Embedding GPU 端到端验证;单元与 serving 测试 14+3 项通过。

## 关联
- #33865(Fixes)
- #44543(非重复:修的是 processor 缓存内部 audio/video 耦合)
- #40986(非重复:Chat/Completion SSE 零 token 错误顺序)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
