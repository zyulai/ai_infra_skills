# PR #22587: [EPD] Optimize the Mooncake backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @LucQueen |
| 合并 | 2026-05-29 |
| 分类 | Overlap / Scheduler |
| Label | high priority, Multi-modal, run-ci |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见 encode_receiver.py 大幅扩展,另有 11 个文件因超长未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/22587 |

## 一句话结论

EPD 分离架构下 Mooncake 后端改为 GPU 直达 GPU 的零拷贝 RDMA 传输视觉 embedding,Qwen3-VL-4B 每 req 8 图时吞吐提升 2.48x、TTFT 降低 8.35x。

## 动机

此前的 EPD(Encoder-Prefill-Decode)分离架构里,Mooncake 后端把 Encoder 侧 ViT 前向产出的 vision embedding 经 CPU 中转:Encoder GPU 上先 `.cpu()` 落地(D2H),经 Mooncake RDMA 传到 Prefill 侧 CPU buffer,再 H2D 拷回 GPU 才能做 LLM prefill。对大输入的多模态请求(如多图 MMMU 任务)而言,D2H → RDMA → H2D 的 CPU 往返带来不必要的延迟和内存带宽开销。

## 优化点

把这条路径改成 GPU-to-GPU 零拷贝 RDMA:(1) ViT 前向后 embedding 不再 `.cpu()`,留在 Encoder GPU 上;(2) Prefill 侧预分配 GPU 接收 buffer 并向 Mooncake engine 注册,经 GPUDirect RDMA 直接写入;(3) 两阶段 encode 协议——`POST /encode` 立即返回元数据(预计算的 embedding_size/dtype/element_size),VIT 前向作为后台 asyncio Task 运行,`POST /send` 时经 event 等待前向完成后触发 RDMA write,实现计算与元数据返回/缓冲区分配的重叠;(4) 新增 `MooncakeEmbeddingPool` 持久化 GPU buffer 池,一次性注册避免每请求 register/deregister,first-fit 分段分配器 + Condition 阻塞等待,slot 释放通过 weakref 绑定到 mm_inputs GC;(5) 接收端直接在 GPU buffer 上按 modality 建 view,跳过 per-part split 与 torch.cat 拷贝。Mooncake 后端同时从 tokenizer 侧收数据路径迁到 scheduler 侧。

## 关键改动

```text
python/sglang/srt/disaggregation/encode_receiver.py: 新增 WaitingImageRDMARequest 与 MooncakeEmbeddingPool(GPU buffer 池)
python/sglang/srt/disaggregation/encode_server.py: 异步 encode_with_mooncake / encode_with_global_cache_mooncake,预计算 embedding 元数据
python/sglang/srt/managers/scheduler.py: init_disaggregation/recv_requests 支持 mooncake 后端
python/sglang/srt/managers/tokenizer_manager.py: mooncake 移交 scheduler 侧,透传 mm_data_mooncake
test/registered/distributed/test_epd_disaggregation.py: 新增 TestEPDDisaggregationMooncake 端到端测试
```

零拷贝接收的核心片段:

```python
# Zero-copy: build per-modality views directly from the pre-registered
# GPU buffer. Skips the per-part split + torch.cat round-trip
if self.embeddings_buffer is not None:
    recv_embedding = _view_pool_buffer_by_modality(
        self.embeddings_buffer, self.recv_embedding_data, self.dtype
    )
```

## 性能声明

PR 正文测试条件:模型 Qwen3-VL-4B-Instruct,部署拓扑 1E3PD on H20,workload 每请求 8 张图,request rate 1.0、32 成功请求。结果:

- 吞吐提升 2.48x:request throughput 从 0.29 req/s 提高到 0.72 req/s;total token throughput 4798.34 → 11967.26 tok/s
- E2E 延迟降低 7.6x:mean E2E 从约 54.5s 降到约 7.1s(P99 82527ms → 14906ms)
- TTFT 降低 8.35x:mean TTFT 从约 51.9s 降到约 6.2s(P99 78007ms → 12032ms)

## 验证与回归风险

新增 `test/registered/distributed/test_epd_disaggregation.py` 的 `TestEPDDisaggregationMooncake`,端到端拉起 encode/prefill/decode 三类服务并跑 MMMU 50 样本断言 accuracy > 0.40(Qwen2.5-VL-3B-Instruct 基线),该测试因需要 RDMA 硬件在 CI 中 skip。风险点:`MMReceiverBase.__init__` 对 mooncake 后端补齐了 scheduler 侧属性并复用 `_init_mm_processor`,重构影响 zmq_to_scheduler 路径;scheduler/tokenizer_manager 的多后端路由逻辑变更可能影响其他 transfer backend 的多模态 EPD 请求;GPU buffer 池大小依赖 `SGLANG_EMBEDDING_POOL_SIZE_MB`,分配超时/超容量会使请求直接失败。

## 关联

同区域优化:EPD 分离架构相关 PR(encode/prefill/decode disaggregation)与多模态调度类 PR 同属 Overlap / Scheduler 分类。PR 正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
