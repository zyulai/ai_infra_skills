# PR #55781: [Frontend][RL] Track HTTP weight operation outcomes and concurrency

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Ronald1995 |
| 合并 | 2026-09-30 |
| 分类 | Frontend / Serving(tier: major) |
| Label | frontend, speculative-decoding, ready, kv-connector, rust |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/55781 |

## 一句话结论

为 RL 权重同步的 5 个 HTTP 端点新增耗时直方图与在途并发 gauge 指标,解决权重传输卡死不可观测的问题。

## 动机

权重同步位于每个 RL step 的关键路径上,但前端没有任何信号反映单次 weight-transfer 操作耗时多久、是否卡住。通用的 `http_requests_total` 按状态码计数了这些路由,但 per-handler 直方图最高只有 1s 桶,且只在请求完成后才记录——一次挂死的 NCCL 传输在指标上完全不可见,无法区分"慢"与"卡死"。

## 优化点

在 Python 与 Rust 两个前端中以相同名称、label、桶定义新增两个 Prometheus 指标:`vllm:rl_weight_update_operation_duration_seconds`(Histogram,桶 0.01–600s,覆盖长传输)与 `vllm:rl_weight_update_operations_in_flight`(Gauge,multiprocess 模式用 `livesum` 跨 API server 进程求和)。`operation` label 区分 `init` / `start` / `start_draft` / `update`(packed 时按 chunk 多次)/ `finish` 五类操作。每个路由只包住 engine 调用本身,被参数校验拒绝的请求不计入;in-flight gauge 在成功、报错、取消三种情况下都会释放(Python 用 context manager 的 finally,Rust 用 Drop guard),duration 也在三种情况下都 observe。Python 侧 collector 懒创建于默认 registry,确保在 multiprocess Prometheus setup 之后注册;不新增请求计数器(已有 `http_requests_total`),与 #52864 的 sleep-mode 指标保持一致。

## 关键改动

```text
vllm/entrypoints/serve/dev/rlhf/metrics.py: WeightOperationMetrics(duration Histogram + in_flight Gauge)与懒加载单例
vllm/entrypoints/serve/dev/rlhf/api_router.py: 5 个权重端点各自包一层 record("...")
rust/src/metrics/src/api_server.rs: WeightOperationRecorder(Drop 时 dec gauge 并 observe duration)
rust/src/server/src/routes/weight_transfer.rs: 5 个 Rust 路由接入 record_weight_operation
```

## 性能声明

无性能数字(本 PR 是可观测性改动,不改变执行路径;PR 自述" No model output path changes, so no accuracy eval")。动机中给出的定量信息仅为:通用 HTTP 直方图 1s 上限不足以覆盖权重传输。E2E 测试在 GB200 上跑通 1 GPU 用例。

## 验证与回归风险

新增 Python 单测 `tests/entrypoints/unit_tests/test_rl_weight_operation_metrics.py`(11 passed:成功/异常/取消路径、5 条路由、被拒请求不记录、懒注册)与 E2E `tests/entrypoints/serve/dev/rlhf/test_weight_operation_metrics.py`(真实 `--api-server-count 2` + IPC 后端,HF trainer 8 次完整同步,校验 per-operation count 与双进程记录);Rust 侧 `vllm-metrics` 与 `vllm-server` 各有 cargo test。风险很小:指标只在权重路由上新增包装,不触碰模型执行路径;需要注意 `WeightOperationMetrics` 全局单例在测试间的注册顺序问题(已用子进程探针测试覆盖)。

## 关联

同区域:#52864(sleep-mode operation 指标,本 PR 的命名与桶设计参照);RL 权重同步链路相关为 frontend/dev/rlhf 路由与 `vllm.distributed.weight_transfer` IPC engine。PR 正文未列出其他前序/后续 PR。
