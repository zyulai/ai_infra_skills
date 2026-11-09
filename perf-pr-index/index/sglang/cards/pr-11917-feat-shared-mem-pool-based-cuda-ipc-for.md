# PR #11917: [FEAT]  Shared mem pool based cuda ipc for multi-modal data transport

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kousakawang |
| 合并 | 2025-11-02 |
| 分类 | Parallelism / Communication (tier: major) |
| Label | high priority, performance, Multi-modal, run-ci, vlm |
| Diff | pack 未提供 diffstat（diff 内容可用：7 个文件，含新增 cuda_ipc_transport_utils.py 314 行） |
| 链接 | https://github.com/sgl-project/sglang/pull/11917 |

## 一句话结论

多模态特征改走共享内存池 + CUDA IPC 传输（env 开关启用），serving 吞吐 3.35 -> 4.69 req/s、mean TTFT 1768.77 -> 757.34 ms，并解决 IPC 显存泄漏。

## 动机

issue #11464 指出：tokenizer 进程向 scheduler 传多模态数据（图像/视频 feature）时，pickle dump + socket 开销大；改用 cuda-ipc handle 可避免该成本。但 PyTorch 限制下，一旦取了 tensor 的 ipc-handle，其显存在进程结束前永不释放——持续运行会让 device memory OOM。早期回收方案（逐 chunk 校验字节）性能差，需要新的设计。

## 优化点

方案是"显存工作区 + 共享内存同步标志"：device 上分配一块 `MmItemMemoryPool`（默认 2GB，`SGLANG_MM_FEATURE_CACHE_MB` 可调），发送时从池里取 chunk 切片，把 mm feature `copy_` 进切片、只发切片的 cuda-ipc handle；接收进程重建后复制到本地 tensor，并用文件锁原子地把共享内存同步标志 +1，使发送侧能据此回收 chunk。池维护 available/occupied 两个 chunk 链表，按 best-fit 切分、按地址合并相邻空闲块；拿不到 chunk 或 handle 分配失败时回退原始 tensor/socket 路径（跨节点也回退 default CPU transport）。代码里注释还记录了此前 recycle_chunks 逐字节校验方案慢的原因，改为 shm 缓冲存同步值。`MM_FEATURE_CACHE_SIZE` 限制传输显存总量；`storage_offset` 修复了切片重建。启用靠 `SGLANG_USE_CUDA_IPC_TRANSPORT=1`。

## 关键改动

```text
python/sglang/srt/utils/cuda_ipc_transport_utils.py: 新增 MmItemMemoryPool / MmItemMemoryChunk / ShmSyncBuffer / CudaIpcTensorTransportProxy
python/sglang/srt/multimodal/processors/base_processor.py: process_and_combine_mm_data 后处理把 feature 拷入池切片并替换为 proxy
python/sglang/srt/managers/mm_utils.py: 反序列化支持 storage_offset；hash_feature 处理 proxy
python/sglang/srt/managers/schedule_batch.py: prepare_for_extend 里 reconstruct_on_target_device
python/sglang/srt/managers/tokenizer_manager.py: _determine_tensor_transport_mode 上移为模块级
test/srt/models/test_vlm_models.py: VLM 测试进程注入 SGLANG_USE_CUDA_IPC_TRANSPORT=1
```

## 性能声明

Serving 对比（128 请求，H20，具体模型/workload 参数 PR 未注明）：Request throughput 3.35 -> 4.69 req/s（+40%）；Output token throughput 858.85 -> 1201.05 tok/s；Total token throughput 4059.40 -> 5676.82 tok/s；Mean TTFT 1768.77 -> 757.34 ms；Median TPOT 5.87 -> 5.54 ms；P99 ITL 39.24 -> 30.21 ms。显存：仅 cuda-ipc 无 workspace 时 serving 后 GPU0 占用 93802MiB（泄漏），加 workspace 后 77652MiB。准确度：Qwen2.5-VL-7B-Instruct MMMU val 开/关 feature 均 mmmu_acc=0.5（n=900），无回归。MMMU 测试时长对比以截图形式给出（正文无数字）。

## 验证与回归风险

VLM 模型测试套件将强制开 `SGLANG_USE_CUDA_IPC_TRANSPORT=1`，MMMU 双跑对比无精度变化。风险：默认关闭、需显式 env 开启；池满或 shm 分配失败回退原路径；chunk 回收依赖接收侧同步标志 +1，若 scheduler 崩溃可能泄漏占用（`__del__` 兜底释放 shm）；`try_to_recycle` 里 tp_size 未初始化时跳过一轮回收；文件锁路径写死 `/tmp/shm_wr_lock.lock`；跨节点自动回退 default transport。

## 关联

resolve #11464（cuda-ipc 传 mm 数据的 OOM 问题来源）。同区域其他优化 PR 正文未列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
