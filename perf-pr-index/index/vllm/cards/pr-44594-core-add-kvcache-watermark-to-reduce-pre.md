# PR #44594: [Core] Add kvcache watermark to reduce preemptions

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2026-06-11 |
| 分类 | KV Cache / Memory（tier: major） |
| Label | performance, ready, v1 |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/44594 |

## 一句话结论

新增 `--watermark`：admit 新请求时预留一定比例 KV cache 空闲块，decode 重负载下抢占 −82%、ITL p99 −56%、吞吐 +5.1%。

## 动机

请求按 admission 时刻所需的 KV cache 被准入：`--scheduler-reserve-full-isl`（默认）会预先保留输入长度，但输出长度未知且不保留。decode 重的 workload（output >> input）在高并发下会在请求还短时过度准入，随着 decode 增长 KV cache 耗尽，scheduler 抢占（recompute）刚准入的请求、之后重新 prefill，如此循环（thrashing）。长输出意味着被抢占请求恢复时要重算约 3500 token，代价很高。watermark 通过保留一块 KV cache 余量让 running 请求增长，避免这种churn。

## 优化点

- 新增 `SchedulerConfig.watermark`（[0.0, 1.0)，默认 0.0 即禁用）：以总 KV cache 块数的比例计算 `watermark_blocks`，作为 admit 等待/被抢占请求时必须保持空闲的最小块数。
- 应用条件精确限定：只作用于 `RequestStatus.WAITING`/`PREEMPTED` 的请求，且仅当本步已有已调度请求（`has_scheduled_reqs=bool(self.running)`）时生效——cache 为空时的第一个新请求不受影响，避免空转。
- 实现上在 `KVCacheManager.allocate_slots` 的两个判断点（`full_sequence_must_fit` 与常规分配）把准入条件从 `num_blocks_to_allocate > free` 改为 `num_blocks_to_allocate + watermark_blocks > free`，不满足则返回 None 推迟准入；与既有的 `reserved_blocks`（in-flight prefill 保留）叠加而非互斥。

## 关键改动

```text
vllm/v1/core/kv_cache_manager.py: watermark_blocks 计算 + allocate_slots 两处准入判断
vllm/v1/core/sched/scheduler.py: 传入 has_scheduled_reqs 与 watermark 配置
vllm/config/scheduler.py + vllm/engine/arg_utils.py: --watermark 配置项（默认 0.0 禁用）
benchmarks/kv_cache_watermark.sh: 可复现 sweep 脚本（含抢占计数抓取与绘图）
tests/v1/core/test_scheduler.py + tests/v1/core/utils.py: 测试调度器固定 watermark=0.0 保证确定性
```

## 性能声明

（Qwen/Qwen2.5-7B-Instruct，TP=1，1x GB200；KV cache 16 GiB = 299,584 tokens，约 1.5 倍平均需求即 near-critical，prefix caching 关闭；`--max-model-len 8192`，`--max-num-seqs 256`，chunked prefill 开启；random 数据集 input ~1000 / output ~5000（range ratio 0.2）、`--ignore-eos`、450 prompts、C=128 饱和。）

| watermark | preempt | out tok/s | ITL p99 (ms) | E2EL p50 (s) |
|---:|---:|---:|---:|---:|
| off | 1065 | 11582 | 40.62 | 51.6 |
| 0.02 | 400 | 12134 | 18.80 | 48.4 |
| 0.05 | 187 | 12167 | 17.70 | 47.9 |
| 0.10 | 168 | 11835 | 17.57 | 48.6 |
| 0.15 | 168 | 11635 | 10.41 | 49.1 |

PR 结论：sweet spot watermark≈0.05——preemptions −82%，ITL p99 −56%，E2EL p50 −7%，throughput +5.1%。注意代价：TTFT p50 从 6.56 s 升至 ~11.2 s（0.02–0.05 档），准入被有意推迟。

## 验证与回归风险

- 验证：新增可复现 benchmark 脚本（从 /metrics 抓 `vllm:num_preemptions_total` 差值计抢占数）；既有调度器测试通过配置 `watermark=0.0` 保持 admission/preemption 机制确定性，diff 未新增 watermark 专项单测。
- 回归风险：默认 0.0 行为不变，零风险面；开启后影响 WAITING/PREEMPTED 请求的准入时机（TTFT p50 明显上升），与 `--scheduler-reserve-full-isl`、async KV-connector load 的 `reserved_blocks` 逻辑叠加，组合语义需注意；文档字符串声明范围 [0.0, 1.0)。

## 关联

- 相关机制：`--scheduler-reserve-full-isl`（scheduler_reserve_full_isl，本 PR 之前已存在的准入预留，watermark 与其互补）。
- 同分类：本批 PR #60695（Mooncake KV connector）、PR #63827（HiSparse async KV load）、PR #51473（KV 容量扩展）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
