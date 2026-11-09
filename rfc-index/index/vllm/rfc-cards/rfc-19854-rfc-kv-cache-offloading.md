# RFC #19854: [RFC]: KV cache offloading

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19854](https://github.com/vllm-project/vllm/issues/19854) |
| 作者 | orozery |
| 创建日期 | 2025-06-19 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 19 · 💬 57 |
| 分类 | KV Cache / Memory |

## 一句话结论
提出新的 OffloadingConnector,为 vLLM v1 提供异步、可插拔后端的 KV cache 卸载(CPU 起步)方案。

## 提案动机
vLLM v1 目前没有官方内置方案把 KV cache 数据从 GPU 内存卸载到其他介质(尤其是 CPU 内存)。已有 RFC #16144 及 PR #13377、#17653 试图解决,但其做法与 V0 类似:在调度器侧扩展 KVCacheManager 支持 CPU offloading,在 worker 侧于 `execute_model` 中同步执行 CPU<->GPU 传输。作者提出一种替代方案以满足更高要求。

## 提出的方案
- 通过新的 OffloadingConnector 实现卸载,对 vLLM core 改动最小
- 支持异步保存(保存完成前不释放 GPU 内存)与异步加载(加载完成前不调度等待请求),与 NixlConnector 类似
- 调度器侧委托抽象 `OffloadingManager` 负责卸载数据的分配与驱逐记账,输出编码进 `KVConnectorMetadata` 发给 worker
- worker 侧 OffloadingConnector 用一组线程异步执行传输(每类传输一个线程,如 GPU->CPU、CPU->GPU,未来还有 CPU->Disk),每个传输请求返回唯一 `job_id` 跟踪完成
- 支持可插拔缓存后端(CPU 后端起步)、可拉取 cache events(insert/evict/access)形成统一 KVCacheEvent 流、支持卸载数据的 LRU 驱逐
- 需要 3 个 core 改动:PR #19555(worker->scheduler 方向的 connector metadata)、PR #19728(引入 `Request.block_hashes` 复用 KVCacheManager 的块哈希)、PR #19737(收集 KV cache events 的 connector API);已有 PR #19848 作为可插拔卸载实现基础

## 优化点与收益
正文未给出量化数字。定性收益:异步保存/加载避免 GPU 内存空闲等待与调度阻塞,缓存事件流与 GPU 缓存统一,LRU 驱逐管理卸载数据。

## 现状与落地
仍开放讨论(keep-open)。作者已开 PR #19848(可插拔卸载基础实现);后续计划在其上叠加调度侧/worker 侧的具体 CPU 实现 PR,最后引入端到端 OffloadingConnector。反馈期一周,CC 了核心维护者。

## 关联
- #16144
- #19555
- #19728
- #19737
- #19848

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
