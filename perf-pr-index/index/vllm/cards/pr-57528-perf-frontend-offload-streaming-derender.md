# PR #57528: [Perf][Frontend] Offload streaming derender detokenization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @KEYS-A15 |
| 合并 | 2026-09-24 |
| 分类 | KV Cache / Memory(major;按 classify.py 原文,改动实为 frontend/streaming 路径) |
| Label | ready |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/57528 |

## 一句话结论

把流式 derender 的增量 detokenization 从 FastAPI 事件循环线程 offload 到 renderer executor,4 worker 下吞吐 +1.2%、derender 中位延迟 +1.6%。

## 动机

修复 #57350:plain streaming chat 与 completion 的 derender 路径在事件循环线程上同步调用 `_detokenize_delta()`。并发流式负载下,这会串行化 derender 处理并延误无关的异步工作。

## 优化点

用 renderer 既有的共享 executor 为 `_detokenize_delta()` 创建异步包装(`make_async(..., executor=renderer._executor)`),`derender_chat_stream()` 与 `derender_completion_stream()` 改为 `await` 该包装;已运行在 renderer executor 内的代码路径保留同步版本。只改变执行位置,不改变 tokenization 语义、stream-state 处理、输出顺序、模型执行或公共 API。与 #50550(流式 reasoning/tool-parser 路径)互不重叠,executor 边界在 #57350 中确认。

## 关键改动

```text
vllm/renderers/online_derenderer.py: 新增 _detokenize_delta_async;两条流式路径 await 它
tests/entrypoints/scale_out/derender/test_derender_stream.py: 回归测试验证 detok 不在事件循环线程
```

## 性能声明

20 并发流、每流 2,048 个单 token chunk、每 run 40,960 次 derender 请求;每 250ms 发一次 1,537-token prompt 的 tokenize 探针;每组配置跑 3 次取中位;对比 `upstream/main@6dc34b6334` 与本 PR `a1de8533fe`。4 个 renderer worker:吞吐 1098.88 → 1112.55 tok/s(+1.2%),derender p50 7.179 → 7.061 ms(+1.6%),p99 基本不变;1 worker:吞吐 -0.2%,中位延迟略改善。注意:4 worker 下 in-load tokenize p99 中位升高(131.9 vs 107.4 ms),但单次 run 区间大量重叠,作者认为可能是 renderer-pool 争用或正常尾部波动,benchmark 无法区分。

## 验证与回归风险

测试:流式 derender 套件 61 passed、1 xfailed(rebase 后;期望失败为 #55195 追踪的既有 parser 路径用例);新增回归测试 `test_streaming_detokenization_runs_off_event_loop` 参数化覆盖两条流式方法,修复前两者失败、修复后通过;pre-commit(mypy/ruff)通过。回归风险:detokenization 每次多一跳 executor 调度,benchmark 未显示 derender 吞吐或 cadence 的实质性回退;不影响模型输出,无需模型评测。

## 关联

修复 #57350;相关路径区分:#50550(reasoning/tool-parser 流式路径);xfailed 用例追踪于 #55195。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
