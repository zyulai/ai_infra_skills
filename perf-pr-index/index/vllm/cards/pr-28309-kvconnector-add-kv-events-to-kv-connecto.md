# PR #28309: [KVConnector] Add KV events to KV Connectors

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hickeyma |
| 合并 | 2025-12-11 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | ready, ci/build, v1, kv-connector |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 4 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/28309 |

## 一句话结论

KV Connector 新增 KV events 通道:连接器(首发 LMCache)产生的 BlockStored 等事件经跨 worker 聚合后交给 vLLM 既有 KV Events publisher 发布。

## 动机

让 KV-cache-aware 路由不仅基于 GPU 中的 KV cache,也能基于 CPU 与本地磁盘中的 KV cache 做更高效的调度决策;是 RFC #20492(KV-Cache Interoperability API Standardization,把 KV Events 形式化为 KV-cache 生命周期事件的公共契约)解决方案的一部分,呼应 #28252 的评审反馈。

## 优化点

定义 `KVConnectorKVEvents` 抽象接口(add_events / aggregate / increment_workers / get_all_events / get_number_of_workers / clear_events),`KVConnectorBase_V1` 增加 `get_kv_connector_kv_cache_events()` 钩子(model runner 每次模型执行后调用,默认返回 None);`KVEventAggregator` 以 Counter 跨 worker 聚合,只返回所有 worker 都上报的公共事件(`BlockStored`/`BlockRemoved` 增加 `__hash__` 以支持计数);`KVOutputAggregator` 聚合各 worker 的 `ModelRunnerOutput` 时合并 `kv_cache_events` 并递增 worker 计数;LMCache connector 实现 `LMCacheKVEvents`:从 lmcache engine 取事件并转换为 `BlockStored`(block_hashes/parent/token_ids/lora_id/block_size/medium,medium 支持 GPU/DISK),经 `update_connector_output`/`take_events` 汇入引擎既有 publisher。

## 关键改动

```text
vllm/distributed/kv_events.py: KVEventAggregator / KVConnectorKVEvents 与事件 __hash__
vllm/distributed/kv_transfer/kv_connector/v1/base.py: get_kv_connector_kv_cache_events 钩子
vllm/distributed/kv_transfer/kv_connector/utils.py: KVOutputAggregator 合并各 worker kv_cache_events
vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py: LMCacheKVEvents 实现与 LMCache 事件转换
tests/v1/kv_connector/unit/test_lmcache_connector.py(新): 聚合/多 worker/重复周期/TP=3 集成场景
```

## 性能声明

PR 未提供性能数据(事件通道/接口改动;正文的 Test Result 一节为空,未附 benchmark)。

## 验证与回归风险

pytest tests/v1/kv_connector/unit/test_lmcache_connector.py(756 行新测试:事件转换与属性保留、聚合只留公共事件、多 worker workflow、真实 KVOutputAggregator TP=3 聚合、重复周期)。风险:KVOutputAggregator 合并要求各 worker 事件容器同类型(assert);连接器不实现接口则无事件(默认 None,行为不变);未配置 connector 的路径不受影响。

## 关联

前序:#28252、RFC #20492;同批 KV Connector 区域:#51052(MoRIIO)、#52921(SimpleCPUOffload)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
