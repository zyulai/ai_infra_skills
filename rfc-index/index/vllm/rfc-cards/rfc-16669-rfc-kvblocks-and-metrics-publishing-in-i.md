# RFC #16669: [RFC]: KVBlocks and Metrics Publishing In Inference Frameworks

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#16669](https://github.com/vllm-project/vllm/issues/16669) |
| 作者 | alec-flowers |
| 创建日期 | 2025-04-15 |
| 状态 | ⚪ closed(关闭于 2025-08-21) |
| labels | RFC, stale |
| 热度 | 👍 10 · 💬 12 |
| 分类 | Frontend / Serving / API |

## 一句话结论
提议向外部推理框架开放 vLLM 引擎状态发布:注入第三方 StatLogger 发布指标,经 KVBlockObserver 观测 KV cache 事件。

## 提案动机
要对分布式 LLM 推理做有效规划与路由,需要把推理引擎的状态发布给分布式系统中的其他组件。例如:本地前缀树增删 block 时发布事件,让 router 汇总所有 worker 的数据构建全局前缀树;实时发布指标,让 router 在某 worker 开始排队请求时做负载均衡。

## 提出的方案
- Metrics:vLLM 已有基于 `StatsLoggerBase` 的指标发布体系(如 `PrometheusStatLogger`);本 RFC 在相关引擎类上新增方法,接受符合 `StatsLoggerBase` 接口的第三方 StatLogger(`add_stats_logger`,或 `AsyncLLM` 构造参数 `stat_loggers`),让包装 vLLM 的推理框架(如 dynamo)注入自定义的指标发布逻辑
- KVEvents:需要更大改动——新增可序列化(msgspec)事件类:`KVCacheEvent` 基类及 `BlockStored`(含 block_hashes、parent_block_hash、token_ids 等)、`BlockRemoved`、`AllBlocksCleared`;在 V1 的 `BlockPool` 中埋点:`cache_full_blocks` 发 `BlockStored`、`_maybe_evict_cached_block` 发 `BlockRemoved`、`reset_prefix_cache` 发 `AllBlocksCleared`;因 BlockPool 由独立进程管理,事件经 zmq 送回客户端,客户端注入 `KVBlockObserver`(`add_kv_block_observer` 或 `AsyncLLM` 构造参数)负责观测并对外发布

## 优化点与收益
正文未给出量化数字(定性:为分布式 router 提供全局前缀树构建与实时负载均衡所需的引擎状态)。

## 现状与落地
已关闭(2025-08-21,label 含 stale)。正文未记录最终落地情况,需查关联 PR。

## 关联
正文未列出(无链接,仅 CC @pavanimajety @robertgshaw2-redhat)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
