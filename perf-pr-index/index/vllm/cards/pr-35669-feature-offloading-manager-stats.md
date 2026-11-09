# PR #35669: Feature/offloading manager stats

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Srinivasoo7 |
| 合并 | 2026-06-10 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | ready, v1, kv-connector |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 3 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/35669 |

## 一句话结论

为 KV offloading 建立统一遥测管线:OffloadingManager 可上报统计,connector 侧改用扁平命名指标(load/store 各自的 bytes/time/size histogram)并聚合到 Prometheus。

## 动机

PR 正文说明:为了与 block-reuse frequency tracking PR 隔离改动,`OffloadingManager` 接口需要一个标准化的方法把统计上报到 scheduler 侧的 `KVConnector` 指标接口。从 diff 可见,改动前 transfer 统计只在 worker 侧的独立 `kv_connector_stats` 容器里按 transfer_type 记录,manager(如 CPU offloading manager)没有上报通道,指标名也耦合在 `OffloadPromMetrics` 构造函数里。本 PR 补齐 manager→scheduler→metrics 的统计链路,为后续基于复用频率的 offloading 策略提供可观测数据。

## 优化点

机制上分四层:(1) `OffloadingManager` 基类新增 `get_stats()`(默认返回 None),并引入 `OffloadingMetricMetadata`(Counter/Gauge/Histogram 三类)与 `OffloadingSpec.build_metric_definitions()`,让每种 offloading spec 声明自己发射的指标。(2) worker 侧新增 `TransferStats`/`DirectionalTransferStats`,按 load/store 方向分别累计 bytes、time 和每次传输 size 列表,记录点从 worker 的独立容器迁入 `OffloadingWorkerMetadata`,随 worker meta 传回 scheduler 并聚合。(3) `OffloadingConnectorStats` 的 payload 改为自描述结构(`types` + `data` 两个 dict):counter 求和、gauge 取最新快照、histogram 拼接样本列表,可跨 IPC 序列化;指标名改为扁平命名(`vllm:kv_offload_load_bytes/_load_time/_load_size/_store_bytes/_store_time/_store_size`),同时保留带 `transfer_type` label 的旧 CPU 指标以兼容迁移期。(4) `CPUOffloadingManager` 直接暴露 `vllm:kv_offload_stores_skipped`,统计因未达到 LRU 复用阈值(`store_threshold`)而被跳过 store 的 block 数。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py: OffloadingConnectorStats 自描述 payload、扁平指标定义、deprecated 指标兼容发射
vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py: 新增 TransferStats/DirectionalTransferStats 记录与聚合
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: scheduler 侧 get_stats() 聚合 worker transfer stats 与 manager stats
vllm/v1/kv_offload/base.py: OffloadingMetricMetadata 元数据与 OffloadingManager.get_stats() 接口
vllm/v1/kv_offload/cpu/manager.py: 暴露 stores_skipped 计数(prepare_store 中统计被阈值过滤的 key 数)
```

关键片段(自描述 stats 结构):

```python
data = {
    _StatsKey.TYPES: {name: "counter"/"gauge"/"histogram", ...},
    _StatsKey.DATA:  {name: value_or_sample_list, ...},
}
# counter 求和、gauge 取最新、histogram extend 样本列表
```

## 性能声明

PR 正文未提供 benchmark 数字与性能对比(此 PR 为可观测性/遥测基建,不直接声称性能收益)。测试结果仅报告 `pytest tests/v1/kv_offload` 与 ruff format/check 通过,未注明任何硬件/模型/workload 条件。

## 验证与回归风险

扩展 `tests/v1/kv_connector/unit/offloading_connector/test_metrics.py`,并在 `test_worker_metadata.py` 新增 `test_aggregate_transfer_stats`(验证 load bytes/time/sizes 跨 worker 聚合)。回归风险主要在指标消费侧:监控 dashboard 需迁移到新的扁平指标名;deprecated 的 `vllm:kv_offload_total_bytes/_total_time/_size` 仅在 spec 为 `CPUOffloadingSpec` 时继续发射,非 CPU offloading 后端将失去旧指标;stats payload 结构变化涉及 scheduler/worker IPC 序列化路径。代码中留有 TODO:待 PR #43877 落地后移除 worker 侧兼容分支。

## 关联

代码注释引用 PR #43877(后续清理依赖);正文说明本 PR 从 block-reuse frequency tracking PR 中拆分而来(该 PR 编号素材未注明)。索引内同区域(KV Cache / KV connector)卡片:#19330(V1 P/D KV load failure recovery)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
