# PR #32104: Add tensor IPC transfer mechanism for multimodal data

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @brandonpelfrey |
| 合并 | 2026-03-21 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | performance, frontend, ready, v1, multi-modality |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动约 14 个文件(含新增 tensor_ipc.py,另有 1 个文件省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/32104 |

## 一句话结论

为多模态张量建立 API Server → CoreEngine 的 torch.multiprocessing 共享内存零拷贝通路,避免 GPU→CPU→GPU 拷贝(视频解码负载多 GPU 扩展的基础设施)。

## 动机

来自 #31925(RFC/PR 拆分):视频解码重的 workload 要做多 GPU 扩展,需要把硬件视频解码结果(以 VRAM 中的 CUDA tensor 形式留在 API Server 进程)传给 CoreEngine 进程。现有路径走 msgpack 序列化,CUDA 张量必须 GPU→CPU 拷贝再序列化,大张量下开销显著。本 PR 是完整方案中独立可评审的一环。

## 优化点

机制是基于 `torch.multiprocessing.Queue` 的共享内存张量传递:新增 `TensorIpcData`/`TensorIpcHandle` 与 `TensorIpcSender`/`TensorIpcReceiver`,sender 把 tensor `share_memory_()` 后连同 (sender_id, message_id, tensor_id) 元数据放入 per-engine 队列,receiver 用 drain-and-buffer 模式(多生产者乱序到达时先缓存再匹配、丢弃 stale tensor)。序列化层 `MsgpackEncoder/MsgpackDecoder` 增加 `oob_tensor_consumer`/`oob_tensor_provider` 钩子:编码时张量交给 sender 走 OOB 通道、消息内只留轻量 handle dict,失败或禁用时回退标准序列化。配置上 `MultiModalConfig.mm_tensor_ipc` 支持 `direct_rpc`(默认)与 `torch_shm` 两模式,`launch_core_engines` 按 DP 引擎建队列并经 handshake 传递索引,`EngineCoreProc`/`DPEngineCoreProc` 的输入解码器接 provider;`abort_requests` 清理孤儿张量。torch_shm 校验:DP/TP/PP>1 不支持(单队列只到 rank 0),且要求 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。仅当多模态处理缓存关闭时生效(cache 会把张量替换成整数)。

## 关键改动

```text
vllm/v1/engine/tensor_ipc.py: TensorIpcSender/TensorIpcReceiver(共享内存队列传输)
vllm/v1/serial_utils.py: OOBTensorConsumer/OOBTensorProvider 钩子接入 msgpack 编解码
vllm/v1/engine/core.py + core_client.py + utils.py: 队列随引擎生命周期创建/handshake/消费
vllm/config/multimodal.py + model.py: mm_tensor_ipc 配置与并行校验
tests/v1/test_tensor_ipc_queue.py: CUDA/CPU IPC、多生产者、缓冲管理、禁用路径测试
```

## 性能声明

本 PR 单独的 A/B 显示性能基本持平(正文:"approximately identical, within noise"):nvidia/cosmos-reason1-7B,H100 + AMD EPYC 9124,TP1,`--mm-processor-cache-gb 0`,sharegpt 数据集 1000 prompts、并发 400、sharegpt-output-len 128——不开 IPC 9.62 req/s / 325.35 output tok/s / Mean TTFT 34909.64 ms;开启后 9.59 req/s / 321.05 output tok/s / Mean TTFT 34538.27 ms(Median TTFT 31583→29643 ms)。零拷贝收益需配合 #31925 的 GPU zero-copy 场景才体现(纯通道基础设施)。

## 验证与回归风险

`tests/v1/test_tensor_ipc_queue.py`(约 943 行):CUDA/CUDA IPC(CUDA 可用时)、多 sender 单 receiver 交错生产、buffer 管理/清理、IPC 禁用;`test_serial_utils.py` 新增非多模态张量字段(prompt_embeds)在 IPC 开启下编解码、None 值、多生产者匹配三个用例(修复了 `_decode_tensor` 误把 handle 当原始数据解码的 bug);依赖既有 CI。风险点:`tensor_data()` 增加 `.cpu()`,小 CPU 张量内联路径不受影响但 CUDA 小张量(<256B 阈值)不再内联;单队列 + 10s put/get timeout 在高负载下可能阻塞或回退;mm cache 开启时通道静默失效,用户配置收益为零。

## 关联

前序/配套:#31925(整体 RFC/PR,多 GPU 视频解码扩展,本 PR 为其拆分组件)。同区域(进程间传输优化):#24078(移除 tokenizer group)、#29826(避免 pageable H2D 传输)。
