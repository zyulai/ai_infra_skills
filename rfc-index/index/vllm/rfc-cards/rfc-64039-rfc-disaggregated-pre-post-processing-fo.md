# RFC #64039: [RFC]: Disaggregated pre/post-processing for pooling IO Processor plugins

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64039](https://github.com/vllm-project/vllm/issues/64039) |
| 作者 | kon-pap |
| 创建日期 | 2026-10-27 |
| 状态 | 🟢 open |
| labels | RFC, kv-connector, pooling |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Disaggregated Serving |

## 一句话结论
把 pooling 模型 IO Processor 的 pre/post 处理拆到无 GPU 的 render 实例,GPU 实例只跑推理,数据经可插拔 connector 传引用。

## 提案动机
IO Processor 插件让 pooling 模型接受/返回任意数据(如 TerraTorch + Prithvi-EO-2.0 地理空间分割)。`pre_process()`(解码、tiling、归一化大 raster)与 `post_process()`(拼 tile、编码输出)CPU 密集,常比 forward 更久;目前三者都挤在拥有 GPU 的 `vllm serve` 进程里:GPU 等 CPU、扩 CPU 容量就得加 GPU 副本、也无法放到更便宜的无 GPU 节点。vLLM 已为生成模型分离 render(`vllm launch render`),并有 KVTransferConfig/ECTransferConfig 的跨实例传数据先例,本文把这两个思路搬到 pooling。

## 提出的方案
- 三阶段请求流:`POST /pooling/preprocess`(render 实例)、`POST /pooling/inference`(`vllm serve`)、`POST /pooling/postprocess`(render 实例);router 在 vLLM 之外,已有可用 proxy 实现。
- 协议传引用不传 payload:新增 DataRef/MetadataRef、GeneratePoolingRequest/Response 等;无 connector 时同端点以 base64 msgpack 内联,便于调试与小部署。
- 新增 `IOTransferConfig`/`--io-transfer-config`(镜像 KVTransferConfig):`io_connector`、`io_role`(io_producer/io_consumer/io_both)等。
- IO connector 框架 `vllm/distributed/io_transfer`:IOConnectorBase 的 put/get/cleanup(data+metadata) 及 async 变体;render 侧阻塞调用走专用线程池(`max_io_threads`);参考后端 IOExampleConnector(共享文件系统如 K8s PVC)与 IOP2PConnector(ZMQ ROUTER/DEALER、pull-based、零拷贝 tensor 帧);复用 MsgpackEncoder/Decoder,无新依赖。
- IOProcessor 新增 `get_request_metadata()`/`update_request_metadata()` no-op 钩子;全 opt-in,现有端点与默认行为不变。

## 优化点与收益
(TerraTorch Prithvi-EO-2.0 分割实测)monolithic 部署推理只占端到端 11–31%,pre 占 31–37%、post 占 33–55%,图像从 1 到 100 tiles 时推理占比递减;80 req/s 下:monolithic(16 workers)约 45 tiles/s(约 40 req/s 即平台化)、mean latency 约 1.7 s、GPU 利用率约 45%;disaggregated(8 pre + 8 post)约 79 tiles/s(仍在扩展)、约 0.34 s、约 84%。

## 现状与落地
open,反馈期两周;已有约 2.6k LOC 的工作实现(六个 commit),计划拆 5 个 PR;遗留限制:仅支持完整三阶段拆分、每 prompt 恰好一个 image 项(否则 400)、IOP2PConnector 高负载性能待调优。

## 关联
正文未列出 RFC/PR 链接(仅引用 `vllm launch render`、KVTransferConfig/ECTransferConfig 等既有机制)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
