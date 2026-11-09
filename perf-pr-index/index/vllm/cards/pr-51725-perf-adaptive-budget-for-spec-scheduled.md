# PR #51725: [Perf] Adaptive budget for spec scheduled token, 55%~65% E2E TTFT Improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-08-11 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | ready, kimi, k3 |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/51725 |

## 一句话结论

调度器对 speculative decoding 的 draft slot 预算从"按 max_num_seqs 全量预扣"改为"按实际调度请求逐个扣减",低并发下 prefill token 预算大幅回升,TTFT 降 55%~65%。

## 动机

开启 spec decode 时,旧逻辑为保证 drafter 可能插入的额外 slot,把 `max_num_scheduled_tokens` 预先减去 `max_num_new_slots_for_drafting * max_num_seqs`。默认 `max_num_seqs=1024`、`max_num_batched_tokens=8192` 时,即使只有 1 个请求,可用调度 token 也被压到 2048,大 prompt 的 prefill 被切碎、TTFT 显著恶化;预留量应与"本 step 实际调度的请求数"成正比,而不是与上限 `max_num_seqs` 成正比。

## 优化点

调度器内新增与 `token_budget` 平行的 `input_budget`(初值为 `max_num_batched_tokens`):每调度一个请求,`input_budget -= num_new_tokens + draft_slots`,即只为本请求预留其 draft slot;每个请求的 `num_new_tokens` 上限变为 `min(token_budget, input_budget - draft_slots)`,`input_budget <= draft_slots` 时停止调度。preemption 恢复时按 `restored` 归还 token 预算与 `restored + draft_slots` 的 input 预算。配置侧 `_set_max_num_scheduled_tokens` 不再从 `max_num_batched_tokens` 中减去 `max_num_new_slots_for_drafting * max_num_seqs`,`max_num_scheduled_tokens` 直接等于 `max_num_batched_tokens`;校验改为仅要求 `max_num_batched_tokens > draft_slots`。效果(PR 给出的调度 token 对照):1 个请求 2048 → 8186,32 → 8000,128 → 7424,1024 并发满载时仍为 2048,即只在低并发时放大预算、高并发行为不变。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: schedule() 引入 input_budget,按请求扣减 num_new_tokens + draft_slots
vllm/config/vllm.py: _set_max_num_scheduled_tokens 不再按 max_num_seqs 预扣 draft slot 预算
tests/v1/core/test_scheduler.py: 新增 parallel_drafting 下 per-request draft slot 预算用例
```

## 性能声明

PR 正文 benchmark(guidellm,synthetic_text prompt_tokens=8000 / output_tokens=1000,concurrent profile,streams 1/4/16;模型 Kimi-K3 TP8 + DSpark dspark k=7 speculative config,`--enable-prefix-caching`,fastsafetensors 加载;结果与 #51726 合并统计,本 PR 单独拆分以便回滚):

| 并发 | 吞吐 | e2e latency | TTFT |
|---|---|---|---|
| 1 | +6.5% | 7.09s → 6.31s(-11.0%) | 1417ms → 491ms(-65.3%) |
| 4 | +14.7% | 14.55s → 11.80s(-18.9%) | 1369ms → 609ms(-55.5%) |
| 16 | +41.5% | 32.29s → 24.81s(-23.1%) | 3218ms → 1217ms(-62.2%) |

精度由 unit tests 覆盖,PR 未附精度 eval 数字。

## 验证与回归风险

`tests/v1/core/test_scheduler.py::test_draft_slots_budgeted_per_scheduled_request`(max_num_seqs=16, max_num_batched_tokens=20, num_speculative_tokens=4, parallel_drafting:两个 10-token 请求调度为 10 与 4,验证 per-request 扣减语义);`create_scheduler` 测试工具新增 `parallel_drafting` 参数。风险:改动触及 v1 调度器热路径的预算逻辑(running 与 waiting 两条队列、preemption 恢复、chunked prefill 与 pooled request 分支),任何 spec decode 配置都受影响;`max_num_scheduled_tokens` 语义变化可能影响依赖旧预留量的下游假设;错误信息与 warning 文案已同步更新。

## 关联

配套/结果合并:#51726(同主题拆分的另一半);同区域 spec decode 调度优化与 DSpark/DSv4 系列卡(pr-56323、pr-54163)。
