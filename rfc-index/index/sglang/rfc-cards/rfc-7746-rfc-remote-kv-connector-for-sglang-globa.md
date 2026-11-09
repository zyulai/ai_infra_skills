# RFC #7746: [RFC] Remote KV Connector for SGLang Global Cache Reuse and PD

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7746](https://github.com/sgl-project/sglang/issues/7746) |
| 作者 | hzh0425 |
| 创建日期 | 2025-07-03 |
| 状态 | ⚪ closed (2025-09-16) |
| labels | high priority |
| 热度 | 👍 16 · 💬 20 |
| 分类 | KV Cache / Memory |

## 一句话结论
提出跨节点 Remote KV Connector 抽象,实现全局 KV cache 复用并与 PD 分离架构集成,减少多轮对话中的重复计算。

## 提案动机
SGLang 缺少全局 PrefixCache 复用能力:多轮对话中已完成推理的 KV cache 无法跨节点复用,相似前缀在不同 worker 节点上被重复处理,造成显著冗余计算与资源浪费。目标包括:定义通用远程连接器接口并实现 3FS Connector、让 Normal/Overlap Scheduler 经 Connector 异步预取/卸载 KV 而不阻塞调度、在多轮场景取得可测量收益、与 PD 分离模式集成。

## 提出的方案
- 定义 `RemoteKVConnector` 抽象接口(`store_kv_async` / `retrieve_kv_async` / `num_external_matched_tokens` / `finish_req` 等),首批实现 3FS Connector 与 localfile Connector,含分块、压缩与错误重试
- `GlobalRemoteStorage` 远程存储引擎(可接 3FS/Mooncake/LMCache 等);`GlobalKVManager` 维护远程全局前缀索引,为可选插件(存储自身支持前缀查询时可省)
- Scheduler 侧 `AsyncKVPrefetchQueue` / `AsyncKVStoreQueue` 异步预取与分块回写,含 TP 一致性完成轮询与资源清理
- PreFetch / Scheduling / PostTransfer 多层队列流水线重叠 KV 传输与计算,隔离 I/O 延迟,不阻塞 Scheduler
- 与 PD 分离集成的两个方向:Prefill 时经 Connector 复用远程缓存、PD 节点间经 Connector 直接传输 KVCache
- 与 HiRadixCache 的差异:独立调度管线、RDMA 直传 GPU 不经 DRAM 中转;专用 Global KVCacheManager 统一管理全局前缀索引,底层存储无需原生前缀查询能力

## 优化点与收益
正文数字(照抄):Global KV Cache Reuse 可实现「~50% TTFT reduction in Qwen-32B 4TP multi-turn dialogues」——测试条件为 Qwen-32B、4TP、多轮对话场景。其余收益为定性:跨节点消除冗余计算、global compute-to-storage conversion、支持 RDMA 直访 HBM 的高吞吐传输。

## 现状与落地
已关闭(2025-09-16)。正文实施计划中 Phase 1–3(接口定义、3FS Connector、队列、Scheduler 集成)各任务勾选为完成,Phase 4 性能评测与稳定性任务未勾选;整体落地情况需查关联 PR。

## 关联
- SGLang PD Disaggregation 内部设计文档(Google Docs,见正文 References)
- 代码路径:`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/mem_cache/hiradix_cache.py`

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
