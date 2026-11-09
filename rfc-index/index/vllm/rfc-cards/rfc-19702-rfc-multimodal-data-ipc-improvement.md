# RFC #19702: [RFC]: Multimodal data IPC improvement

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19702](https://github.com/vllm-project/vllm/issues/19702) |
| 作者 | dongluw |
| 创建日期 | 2025-06-16 |
| 状态 | ⚪ closed(关闭于 2025-09-12) |
| labels | RFC |
| 热度 | 👍 12 · 💬 0 |
| 分类 | Other |

## 一句话结论
提议用共享内存方式重构 vLLM 进程间通信,减少多模态数据经 socket 与 pickle 传输的开销。

## 提案动机
当前 vLLM 的进程间通信在某些场景下开销可观:内部视觉模型在 TP>1 设置下的 profiling 显示,GPU 在 engine 与 worker 通信期间处于空闲。主要开销来自两部分:engine 与 worker 进程间经 socket 的 IPC,以及基于 pickle 的序列化/反序列化。此外多模态数据还存在多余传输:先从 P0 传到 engine,再由 engine 传给 worker。

## 提出的方案
与 @ywang96、@njhill 初步讨论后,针对三类通信开销分三步解决:
1. **engine↔worker IPC 走共享内存**:现有 `ShmRingBuffer` 只支持固定大小 chunk,受 chunk 大小限制,默认仅在 mm 数据小于 16MB 时启用(PR #19242),否则退回较慢的 socket;拟新增/重新设计支持变长 mm 数据的共享内存 buffer 实现。
2. **跳过 mm 数据 (反)序列化**:在上述基础上,RPC 调用只保留 mm_hashes,不序列化 mm 数据本身;类似 P0 缓存命中时在 engine 进程把 mm data 置 None、在 worker 进程从共享内存 buffer 恢复。前提假设:mm 数据只含 numpy/torch 张量或其他易序列化类型。
3. **消除多余传输**:理想情况下用同一共享内存 buffer 替换 `MirroredProcessingCache`,避免 mm 数据在所有进程之间的额外拷贝。

## 优化点与收益
正文未给出量化数字。收益定位:消除 TP>1 时 engine-worker 通信期间 GPU 空闲,减少 socket IPC 与 pickle 序列化/反序列化开销,并省去多模态数据在 P0→engine→worker 的重复传输。

## 现状与落地
已关闭(2025-09-12)。正文未说明实现进度,落地情况需查关联 PR。

## 关联
- https://github.com/vllm-project/vllm/issues/16626(类似问题)
- https://github.com/vllm-project/vllm/pull/19242
- [ShmRingBuffer 源码](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/shm_broadcast.py#L68)
- [mm_input_cache.py 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/mm_input_cache.py#L58)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
