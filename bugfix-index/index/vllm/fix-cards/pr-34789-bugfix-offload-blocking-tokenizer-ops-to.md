# PR #34789: [Bugfix] Offload blocking tokenizer ops to shared thread pool to unblock event loop

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34789](https://github.com/vllm-project/vllm/pull/34789) |
| 作者 | scyyh11 |
| 合并日期 | 2026-03-27 |
| labels | bug, ready, v1 |
| 热度 | 👍 0 · 💬 74 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
多模态预处理与 chat 模板渲染同步阻塞 asyncio 事件循环致 /health 无响应;改为卸载到共享线程池。

## 问题现象
高并发下,多模态请求预处理(base64 解码、图像变换、HF processor 操作)与 chat 模板渲染这类同步 CPU 密集操作阻塞 asyncio 事件循环,导致 `/health`、`/v1/models`、`/metrics` 无响应:P95 延迟 >200ms,尖峰超 1s。PaddleOCR-VL-1.5 高并发基准(500 张 OmniDocBench 真实文档图、300 并发、`--max-num-batched-tokens 131072`)中 /health 中位 222.44ms、P99 1,641.53ms;修复后中位 0.70ms、P99 18.88ms(318x/87x 改善),吞吐 +3.7%、TTFT mean -5.9%。text-only(Llama-3.1-8B-Instruct)与多模态(Qwen2.5-VL-7B-Instruct)基准均无回归;各测试中 "Already borrowed" 错误为 0。

## 根因
正文披露:上述同步 CPU-bound 操作直接在 asyncio 事件循环上执行,高并发下阻塞事件循环使其失去响应。正文并指出关键收益来自把预处理卸载(offload)出事件循环而非并行化——`--preprocessing-thread-pool-workers` 取 1/2/4 表现一致,默认 workers=1 即足够。

## 修复方案
- 在 `BaseRenderer` 新增共享 `ThreadPoolExecutor`,大小由新参数 `--renderer-num-workers`(`renderer_num_workers`,默认 1)控制;多模态预处理始终卸载到该线程池。
- `HfRenderer`、`MistralRenderer`、`DeepseekV32Renderer`、`Grok2Renderer` 的 chat 模板渲染经 `make_async` 包装到共享执行器;`MistralRenderer` 原独立 `ThreadPoolExecutor` 并入共享池。
- `AsyncMicrobatchTokenizer` 接受外部 executor;新增 `process_for_engine_async` 等 async 路径,`render_cmpl_async`/`render_chat_async` 改为 `asyncio.gather` 并发处理。
- `clear_mm_cache` 经共享执行器串行化(`clear_mm_cache_async`),避免与并发 `process_inputs` 在 `mm_processor_cache` 上竞态;`shutdown()` 负责关闭执行器。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:按 diff,涉及 `vllm/renderers/*`(base/hf/mistral/deepseek_v32/grok2)、`vllm/utils/async_utils.py`、`vllm/v1/engine/async_llm.py` 的 `reset_mm_cache`、`vllm/config/model.py` 与 `arg_utils.py` 新增参数;全部 tokenize/渲染 async 调用路径改走线程池(默认单 worker,串行执行);正文基准显示 text-only 吞吐 -0.2%、多模态 +0.1%,均在噪声内。

## 关联
- [PR #34789 评论(workers 对比一致性依据)](https://github.com/vllm-project/vllm/pull/34789#issuecomment-3055653700)
- diff 注释中引用 #36557(mm processor 收到深拷贝 tokenizer,可安全并发)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
