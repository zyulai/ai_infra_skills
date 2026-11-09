# PR #17026: feat: Priority-based scheduling optimization (including default priority, preemption toggle, priority-based metrics, etc.)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zhuxinjie-nz |
| 合并 | 2026-03-04 |
| 分类 | Overlap / Scheduler |
| Label | documentation, amd, run-ci |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/17026 |

## 一句话结论

完善基于优先级的调度:默认优先级、抢占开关与按优先级拆分的队列/延迟指标。

## 动机

在实际使用 priority-based scheduling 时发现三个问题:原逻辑不支持设置默认优先级(未显式带 priority 的请求无法归入既定优先级);抢占逻辑在作者的特定业务场景下有风险,需要可配置的开关;缺少按优先级维度的指标,无法做基于优先级调度的性能分析。

## 优化点

三块改动。其一,默认优先级:新增 `--default-priority-value`,tokenizer_manager 在 generate_request 入口对 priority 为 None 的请求统一填入默认值,避免 Prometheus label 出现 "None"。其二,抢占开关:原 `try_preemption`(随 enable_priority_scheduling 隐式全开)重命名为 `enable_priority_preemption`,新增 `--disable-priority-preemption`;默认行为不变(优先级调度仍默认开抢占),但业务方可以关掉;DLLM(diffusion LLM)调度 mixin 中对应分支同步改名。其三,优先级指标:引入 QueueCount 数据类(total + by_priority 分桶),SchedulerStats 的 num_running_reqs/num_queue_reqs 及 PD 分离的四个队列计数全部改为 QueueCount;metrics_collector 的 _log_gauge 在 priority 调度下把总数记在 `priority=""`、各优先级记在 `priority="<int>"`,并用 _known_priorities 集合维持已出现过的优先级时间序列(值为 0 也持续输出);tokenizer_manager 侧为 TTFT/e2e 等 histogram 追加 priority label;PrefillStats 改用 from_adder 工厂携带 QueueCount。

## 关键改动

```text
python/sglang/srt/managers/tokenizer_manager.py: _set_default_priority 默认优先级注入 + 请求指标加 priority label
python/sglang/srt/server_args.py: 新增 --disable-priority-preemption 与 --default-priority-value
python/sglang/srt/observability/metrics_collector.py: QueueCount 与 _log_gauge 的 per-priority 输出
python/sglang/srt/managers/scheduler.py: try_preemption 改名 enable_priority_preemption 并受开关控制
test/registered/metrics/test_priority_metrics.py: 新增优先级指标 e2e 测试
```

## 性能声明

PR 正文无数字化 benchmark 表格(Benchmarking 一节仅附图):文字说明为——从并发 40 起,请求开始排队,低优先级请求的排队数量显著高于高优先级请求,符合预期;正文未提供具体数值,素材中也未检测到 perf claim。

## 验证与回归风险

新增 test/registered/metrics/test_priority_metrics.py(Qwen/Qwen3-0.6B,--enable-metrics --enable-priority-scheduling --default-priority-value 0):验证 gauge(num_running_reqs/num_queue_reqs)含 priority="" 总量样本、histogram(TTFT/e2e latency)含非空 priority label、未显式给 priority 的请求落到默认值 "0"。回归面:QueueCount 改变了 SchedulerStats 字段类型,所有消费方(PD 分离队列、KvMetrics、utilization 计算)已同步取 .total;开启优先级调度时 Prometheus 指标会新增 priority 标签维度(series 基数随不同优先级数量增长,正文注释提醒应设置默认优先级避免 "None" label);关闭开关时指标路径与原先一致。

## 关联

前序:enable_priority_scheduling 及 priority_scheduling_preemption_threshold 等既有优先级调度参数(基础实现 PR 编号素材未给出);同分类为 Overlap / Scheduler 的其他调度优化,素材未给出编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
