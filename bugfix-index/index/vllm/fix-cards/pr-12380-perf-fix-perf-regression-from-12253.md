# PR #12380: [perf] fix perf regression from #12253

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#12380](https://github.com/vllm-project/vllm/pull/12380) |
| 作者 | youkaichao |
| 合并日期 | 2025-01-24 |
| labels | ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#12253 把 `decode_only` 误解为模型是否 decoder-only 而在 `__init__` 只初始化一次,导致后续批次沿用上一批的值、cudagraph 判定失效;移入每个批次执行的 `prepare` 修复,延迟回落约 11%。

## 问题现象
#12253 合入后出现性能回归。作者用 `python benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --load-format dummy` 实测:main 分支 Avg latency 1.1251 秒,本 PR 后降到 1.0009 秒(50% percentile 1.1259 → 1.0010 秒)。

## 根因
PR 正文披露(作者自述):做 #12253 时误以为 `self.decode_only` 表示该模型是否 decoder-only 模型(静态属性),而它实际表示「当前批次是否为纯 decode 批次」(据此可使用 cudagraph)。由于该字段在 `__init__` 中只初始化一次,后续每个批次都沿用上一个批次的值;一旦某个含 prefill 的批次把它置为 `False`,之后所有批次都被误判为非 decode-only,cudagraph 无法使用,造成性能回归。

## 修复方案
`vllm/worker/model_runner.py`:把 `self.decode_only = True` 从 `__init__` 移入每个批次都会执行的 `prepare` 函数开头,使其逐批次重置:

```diff
-        self.decode_only = True
```

```diff
         self.finished_requests_ids = finished_requests_ids
 
+        # if the current batch is decode-only.
+        # will be set to False if there is any non-decode request.
+        self.decode_only = True
+
         # Intermediate data (data in CPU before going to GPU) for
         # the current sequence group.
```

## 影响与回归风险
- 影响窗口:自 #12253 合入起至本修复前的版本受影响。
- 回归面:仅改动 `vllm/worker/model_runner.py` 中 `decode_only` 的初始化位置;`prepare` 每批次重置后,含 prefill 的批次仍会被后续逻辑置 `False`,语义与字段本意一致。正文未报告其他路径受影响。

## 关联
- #12253(引入回归的原 PR,正文引用,未给出链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
