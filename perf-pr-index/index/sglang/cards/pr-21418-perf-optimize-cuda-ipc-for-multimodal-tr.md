# PR #21418: [Perf] Optimize CUDA IPC for multimodal transfer by caching IPC pool handles

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @saatwiknagpal |
| 合并 | 2026-03-29 |
| 分类 | KV Cache / Memory |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/21418 |

## 一句话结论

缓存 multimodal CUDA IPC pool 的已打开 storage,消除 mm_loop 每轮重复 open handle,吞吐 +16.2%、TTFT -13.2%。

## 动机

multimodal 模型启用 CUDA IPC 传输(`SGLANG_USE_CUDA_IPC_TRANSPORT=1`)时,每个 scheduler iteration 都会通过 `torch.UntypedStorage._new_shared_cuda()` 为同一个长生命周期 memory pool 重新打开 IPC handle。底层 `cuIpcOpenMemHandle` 驱动调用虽有引用计数,但 PyTorch wrapper 每次调用都有 storage 对象创建、event 管理与 Python 级 bookkeeping 的显著开销(引用 pytorch/pytorch#161481)。而 pool 在启动时只分配一次,重复 open 完全冗余。该开销位于 scheduler 关键路径(`mm_loop`)上,会拖慢所有 in-flight 请求的 prefill 与 decode iteration。

## 优化点

生产端:`MmItemMemoryPool` 初始化时对整个 pool 一次性取 IPC handle(`storage._share_cuda_()`),而非每 slice 传输时各自携带 handle;proxy 中改传 `pool_ipc_handle + pool_byte_offset + pool_device_index`。消费端:新增进程级 `_pool_storage_cache` dict,以 `(device_index, handle tuple)` 为键缓存已打开的 `UntypedStorage`,用 double-checked locking 保证线程安全;重建张量时直接在缓存的 pool storage 上按 byte offset 切片再 `set_` 出目标张量,不再重复 open。健壮性:从缓存路径反序列化失败时,作废该 cache entry 并走 uncached 路径重试(可把新 storage 重新入缓存);非 pool 的原始 per-handle 路径保留为回退。功能由 `SGLANG_USE_IPC_POOL_HANDLE_CACHE` 环境变量控制,默认关闭。

## 关键改动

```text
python/sglang/srt/utils/cuda_ipc_transport_utils.py: 新增 _pool_storage_cache 与缓存读写/作废函数,消费端切片重建
python/sglang/srt/multimodal/processors/base_processor.py: proxy 携带 pool handle + byte offset
python/sglang/srt/environ.py: 新增 SGLANG_USE_IPC_POOL_HANDLE_CACHE 开关(默认 False)
```

```python
storage = _pool_handle_cache_get_or_open(cache_key, ipc_extra["pool_handle"])
slice_storage = storage[offset : offset + nbytes]
slice_tensor = torch.empty(0, dtype=dtype, device=target_device).set_(
    slice_storage, storage_offset=..., size=shape, stride=stride)
```

## 性能声明

Qwen/Qwen2.5-VL-3B-Instruct,1xH100,16 tokenizer workers,mixed chunk ON,CUDA IPC ON;bench_serving image 数据集 3000 prompts,input 128 / output 32,concurrency 50,每配置 10 次取中位数:

- Request throughput:22.2 → 25.7 req/s(+16.2%)
- TTFT mean:965 → 838 ms(-13.2%;p99 2058 → 1819 ms)
- TPOT mean:72 → 60 ms(-17.2%)
- ITL mean:67 → 56 ms(-16.2%;p99 627 → 556 ms)
- E2E latency mean:1979 → 1768 ms(-10.6%)
- 错误数两组相同(2-3/3000)

## 验证与回归风险

新增 `test/registered/utils/test_cuda_ipc_transport_utils.py`:cache reuse(同 pool 两个 proxy 只触发一次 `_new_shared_cuda`)、stale entry recovery(投毒条目被作废并 uncached 重试)、cross-pool isolation。正确性:相同输入(seed 2710)3000 请求对比,52% 输出 byte-identical,其余差异归因于 c=50 非确定性调度,输出长度分布一致,无损坏输出。风险:进程级缓存 GPU storage 的生命周期依赖异常驱动的 invalidate-retry;与 pool recycle(`SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC`)的交互是主要失效场景;默认关闭,行为仅在开关开启时改变。

## 关联

同区域:ViT piecewise CUDA graph(PR #15320)的 benchmark 同样依赖 `SGLANG_USE_CUDA_IPC_TRANSPORT=1` 的 multimodal 传输路径。同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
