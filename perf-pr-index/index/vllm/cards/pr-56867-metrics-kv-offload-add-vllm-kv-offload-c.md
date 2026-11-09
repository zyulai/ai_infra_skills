# PR #56867: [Metrics][KV Offload] Add vllm:kv_offload_config_info - generic info metrics for offloading the offloading connector

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amirfr3 |
| 合并 | 2026-10-14 |
| 分类 | KV Cache / Memory \| tier: major |
| Label | documentation, ready, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 4 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/56867 |

## 一句话结论

新增 `vllm:kv_offload_config_info` gauge，让 KV offload 各 tier 的静态配置可由 /metrics 读出。

## 动机

PR 正文 Purpose 说明，offloading spec 持有静态配置却没有任何 metric 上报：CPU cache 只导出 usage（`vllm:kv_offload_cpu_cache_usage_perc` 及 read/write 变体），secondary tier 什么都不导出，运维无法从 `/metrics` 读到 tier 的 capacity、mount path 或 eviction policy；每新增一个事实都要自带 metric、label list 与 emission call。本 PR 新增常量值 1 的 gauge `vllm:kv_offload_config_info` 与两个 hook：spec 声明 label 名、manager 填值。

## 优化点

`OffloadingSpec.config_info_keys(extra_config)` 声明 label 名，`OffloadingManager.config_info()` 返回 mapping 序列，`SecondaryTierManager` 同样带这两个方法且默认返回 `[{}]`。一个 tier 一个 series，用 `tier` label 区分（值来自 `TieringMetricsTracker.tier_label()`，与 tiering counters/histograms 一致；常量 `TIER_LABEL` 在 `vllm/v1/kv_offload/tiering/base.py`）。`TieringOffloadingSpec.config_info_keys()` 声明各配置 tier 名字的并集加 `tier`，manager 为每个 mapping 补 `tier` 与它不拥有的 name 的空值（同一 metric 的每个 series 都带全部 label）。`ConfigInfo` dataclass（`vllm/v1/kv_offload/base.py`）字段名即 label 名；`CPUOffloadingInfo` 发布 `cpu_num_chunks`，`ExampleTierInfo` 发布 `example_info`。信息经 `OffloadingConnectorScheduler.get_stats()` 写入 `_StatsKey.INFO`（`_info_sent` 保证只发一次），由 `_observe_info()` 对齐 label 并每个 mapping 报一个 series。名称不一致时 spec 为权威：声明未填为 `""`、填了未声明由前端丢弃，每 engine 一条 `warning_once`。另修正 `_create_metric` 未传 `multiprocess_mode` 导致 `--api-server-count > 1` 下 per-engine 常量被 `sum` 放大的问题，把默认 `mostrecent` 放到 `OffloadingGaugeMetadata`（仅 Gauge 可用）。

## 关键改动

```text
vllm/v1/kv_offload/base.py: 新增 ConfigInfo dataclass 与 OffloadingManager.config_info() 默认实现
vllm/v1/kv_offload/tiering/base.py: 定义 TIER_LABEL 常量，与 tiering metrics 共用
vllm/v1/kv_offload/tiering/spec.py、manager.py: config_info_keys() 声明 name 并集+tier，manager 为每个 mapping 补 tier label 与空值
vllm/v1/kv_offload/cpu/manager.py、spec.py、common.py、tiering/example/manager.py: CPUOffloadingManager 发布 cpu_num_chunks，example tier 发布 example_info
vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py、scheduler.py、docs/features/kv_offloading_usage.md: _observe_info() 上报 series、get_stats() 写 INFO，并更新文档
```

```python
class OffloadingSpec(ABC):
    @classmethod
    def config_info_keys(cls, extra_config: dict[str, Any]) -> tuple[str, ...]:
        return ()


class OffloadingManager(ABC):
    def config_info(self) -> Sequence[Mapping[str, str | int | float | bool]]:
        return [{}]
```

（摘自 PR 正文：spec 声明名字、manager 填值的两个 hook。）

## 性能声明

PR 正文未给出定量性能数据。正文为 metrics 设计说明，无 benchmark 数字，pack 的 perf claims regex 未检测到；PR 未注明测试条件。

## 验证与回归风险

PR 引入/依赖的测试：正文说明 docstrings 要求实现者在测试中 assert 名称一致性以由 CI 捕获（spec 声明但无 tier 填充的情形），可见 diff 含 `tests/v1/kv_connector/unit/offloading_connector/test_metrics.py`；正文未给出具体测试命令或结果数字。回归风险：正文明确 `multiprocess_mode="mostrecent"` 修正了现有七个 offloading gauges 在多 frontend 下被 `sum` 放大的问题；`_observe_info()` 不得抛异常——`_get_prometheus_metric()` 抛异常会终止 output handler，使所有 in-flight 请求永久等待；`KVConnectorLogging.log()` 改为跳过空 summary。可能受影响路径：offloading gauges、`/metrics` 输出、`--api-server-count > 1` 部署、`KVCRSecondaryTierManager`（唯一改写 metric name 的 tier）。

## 关联

正文 "Relation to #53902"：PR #53902 会为 `CPUOffloadingInfo` 增加 `cpu_blocks_per_chunk`、`cpu_kv_bytes_per_chunk`、`cpu_capacity_tokens_at_max_len` 三个字段；此处先完成 tiering compose，使 out-of-tree secondary tier 即可得到 series。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
