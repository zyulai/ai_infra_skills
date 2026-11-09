# PR #34936: [UX] Add `--performance-mode {balanced,interactivity,throughput}`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-02-26 |
| 分类 | Other(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 vllm/config/vllm.py 与 vllm/engine/arg_utils.py |
| 链接 | https://github.com/vllm-project/vllm/pull/34936 |

## 一句话结论

新增 --performance-mode {balanced,interactivity,throughput} 旗标声明负载意图:interactivity 捕获 step-1..32 细粒度 CUDA graph,throughput 翻倍默认 batch 上限。

## 动机

此前用户要表达"要低延迟还是要吞吐"必须逐项调 cudagraph 捕获尺寸、max_num_seqs 等参数。PR 引入一个顶层旗标让用户声明 workload 意图,默认 balanced 完全保持现有行为;正文还列举未来可挂接的方向:调度器特化 batch size/平滑度、捕获更多 graph、在无法运行时切换的 oracle 里选内核(triton vs deepgemm moe)、扩大 api server 数/stream interval 降低 CPU 瓶颈、按延迟或吞吐开 fusion。

## 优化点

VllmConfig 新增 performance_mode(Literal["balanced","interactivity","throughput"])字段与 --performance-mode CLI 参数,非 balanced 时打一条 info 日志。_set_cudagraph_sizes 中 interactivity 模式把捕获尺寸设为 1..min(max_cudagraph_capture_size, 32) 逐 1 捕获(细粒度 CUDA graph、最小 padding 开销);throughput 模式在用户未显式设置时把 max_num_batched_tokens 与 max_num_seqs 各翻倍。另对 cudagraph_capture_sizes 做去重排序,防止 interactivity 的小步长与既有阶梯重叠产生重复项。

## 关键改动

```text
vllm/config/vllm.py: PerformanceMode 类型、VllmConfig.performance_mode 字段、
  interactivity 模式的 cudagraph 捕获尺寸(1..32 逐 1)逻辑
vllm/engine/arg_utils.py: --performance-mode CLI 参数;throughput 模式下默认
  max_num_batched_tokens / max_num_seqs 翻倍
```

## 性能声明

PR 未给出 benchmark 数字;正文仅说明 interactivity 捕获 step-1 到 32 的细粒度 CUDA graph、throughput 模式翻倍默认 batch 限制。PR 未注明测试条件(无 perf 数据)。

## 验证与回归风险

PR 的 Test Plan/Test Result 为空,未列测试。默认 balanced 不改变行为;风险集中在两个新模式:interactivity 逐尺寸捕获显著增加 graph 数量与显存占用,throughput 翻倍批量上限会放大峰值显存。cudagraph_capture_sizes 的去重排序对所有未显式配置的路径生效。

## 关联

无(pack 未列出关联 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
