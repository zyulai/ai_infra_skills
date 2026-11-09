# PR #14123: [VLM] Boost Memory Pool based CUDA IPC

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-12-01 |
| 分类 | KV Cache / Memory |
| Label | performance, Multi-modal, ready-to-merge, run-ci, vlm |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/14123 |

## 一句话结论

把 MmItemMemoryPool 的 chunk 回收/合并改为独立后台线程,VLM CUDA IPC 传输高并发下性能 +10%,concurrency=10 TTFT 快 8 倍。

## 动机

排查 issue #13940:`SGLANG_USE_CUDA_IPC_TRANSPORT=1` 且 TP>1 时,高流量 VLM 服务出现 OOM。实际不是 GPU 内存泄漏,而是 #11917 引入的 MmItemMemoryPool 不及时回收:`try_to_recycle` 只在 `process_and_combine_mm_data → return_a_slice_tensor_with_flag → recycle_chunks → try_to_recycle` 这条同步链上被调用,且要求收集齐所有 TP rank 的 ref-cnt 才算回收。高负载下各 TP rank 处理速度有 skew,慢 rank 未回写时 recycler 认为该 chunk 仍被占用,转而新分配 chunk,GPU 占用持续增长且回收几乎总是 miss。

## 优化点

在 `MmItemMemoryPool` 内起一个 daemon 后台线程 `_recycle_loop`,按 `MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL`(默认 0.05 s,可用 `SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC` 调整)周期性执行 `recycle_chunks` 与 `merge_chunks`,把回收从请求处理关键路径中剥离;`return_a_slice_tensor_with_flag` 不再内联做 recycle/merge,只持锁查询 available chunk。新增 `threading.Lock` 保护 pool 的 chunk 列表状态。另修一处 `try_to_recycle` 语义:`get_global_server_args` 未初始化时直接 `return False`,不再用 `tp_num=-1` 参与比较(旧写法在 flag==−1 时会误清零)。默认 `MM_FEATURE_CACHE_SIZE` 从 2 GB 提到 4 GB。

## 关键改动

```text
python/sglang/srt/utils/cuda_ipc_transport_utils.py: MmItemMemoryPool 增加后台回收线程、锁,修正 try_to_recycle 异常路径
python/sglang/srt/multimodal/processors/base_processor.py: 构造 MmItemMemoryPool 时传入 recycle_interval
python/sglang/srt/utils/common.py: 新增 get_float_env_var
```

关键机制(后台周期回收):

```python
def _recycle_loop(self):
    while not self._stop_recycler:
        with self._lock:
            self.recycle_chunks()
            self.merge_chunks()
        time.sleep(self._recycle_interval)
```

## 性能声明

PR 正文 benchmark(bench_serving image 数据集,100 prompts,每 prompt 7 张 jpeg 1120x700,random input/output len 100;Qwen3-VL-8B-Instruct,TP=2,SGLANG_USE_CUDA_IPC_TRANSPORT=1,SGLANG_MM_FEATURE_CACHE_MB=4096,SGLANG_VLM_CACHE_SIZE_MB=512,flashinfer + fa3):

- max-concurrency=100:对比基线 #11917,input token throughput 7502.13 → 8235.44 tok/s,Mean TTFT 41101.45 → 35076.29 ms,Request throughput 1.36 → 1.49 req/s;自述 "performance improved 10%"。
- max-concurrency=10,对比 main(legacy):自述 "speed-up 8x",Mean TTFT 465.59 ms vs 4222.62 ms;input throughput 42778.64 vs 7900.68 tok/s。

## 验证与回归风险

PR 未注明新增单元测试;验证依赖上述 bench_serving 对比。风险点:后台线程与请求路径并发访问 pool,靠单一 `threading.Lock` 串行化,若 chunk 的 sync_flag(ref-cnt)写入路径在锁外,存在竞态可能;回收间隔默认 0.05 s 可调,过小增加锁竞争、过大延后回收;默认 cache 从 2 GB 提到 4 GB,显存紧张时需用 `SGLANG_MM_FEATURE_CACHE_MB` 显式调小。

## 关联

前序:#11917(memory pool based CUDA IPC 机制引入)。同区域:#21418(optimize CUDA IPC for multimodal)。issue 关联:#13940。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
