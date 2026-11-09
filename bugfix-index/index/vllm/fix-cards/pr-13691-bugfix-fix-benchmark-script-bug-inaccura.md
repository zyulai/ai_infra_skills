# PR #13691: [Bugfix] Fix benchmark script bug: inaccurate stats for vllm backend when max_model_len < input_len + output_len

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#13691](https://github.com/vllm-project/vllm/pull/13691) |
| 作者 | WangErXiao |
| 合并日期 | 2025-02-22 |
| labels | ready, structured-output |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

benchmark 脚本在 max_model_len 不足时统计失真,现启动前加断言提前报错。

## 问题现象

使用 vllm 后端的 benchmark 脚本(benchmark_guided、benchmark_latency、benchmark_prioritization、benchmark_throughput)时,若 `max_model_len` 小于 `prompt_len + expected_output_len`,请求可能提前终止、实际生成长度达不到 `expected_output_len`,导致吞吐/延迟统计结果不准确。正文未给出具体报错日志,影响面为所有用这些脚本做性能测试的场景。

## 根因

按正文披露:当 `max_model_len` 小于 `prompt_len + expected_output_len` 时,请求可能提前结束而达不到期望输出长度;而脚本的总生成 token 数是按「`expected_output_len` × prompt 数量」计算的,并不检查实际生成量,于是统计被系统性高估/失真。脚本此前没有任何前置校验,错误配置会静默跑完并产出失真数据。(由 diff 推断)改动方式暗示作者选择在 `run_vllm`/`run_vllm_async` 入口处以断言快速失败,而不是在统计侧做补偿。

## 修复方案

在四个 benchmark 脚本的 vllm 后端运行入口增加断言,确保 `max_model_len >= prompt_len + expected_output_len`(或 `input_len + output_len`),不满足时给出明确报错信息 "Please ensure that max_model_len is greater than the sum of prompt_len and expected_output_len for all requests."。改动文件:

- `benchmarks/benchmark_guided.py`:`run_vllm` 同步路径用 `llm.llm_engine.model_config.max_model_len`,异步路径 `run_vllm_async` 用 `llm.model_config.max_model_len`,逐条 request 校验。
- `benchmarks/benchmark_latency.py`:`main` 中用 `args.input_len + args.output_len` 校验。
- `benchmarks/benchmark_prioritization.py`:`run_vllm` 中按 `request[1] + request[2]` 校验;另将随机优先级生成抽为 `get_random_flag()`,并在合成数据路径中为每条 request 附上优先级字段(顺带重构)。
- `benchmarks/benchmark_throughput.py`:同步/异步两个入口均逐条校验。

关键 diff 摘录(benchmark_throughput.py):

```diff
+    assert all(
+        llm.llm_engine.model_config.max_model_len >= (
+            request.prompt_len + request.expected_output_len)
+        for request in requests), (
+            "Please ensure that max_model_len is greater than the sum of"
+            " prompt_len and expected_output_len for all requests.")
```

## 影响与回归风险

- 影响窗口:正文未说明具体受影响版本范围,不写。
- 回归面:仅影响 `benchmarks/` 目录下四个基准脚本,不触及引擎代码;若用户传入的 `input_len + output_len` 超过 `max_model_len`,脚本会从「静默产出失真数据」变为「断言报错退出」,属于预期行为收紧。`benchmark_prioritization.py` 的随机优先级抽取为纯等价重构,但改变了合成请求数据结构(三元组变四元组),需与脚本内其他引用保持一致。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
