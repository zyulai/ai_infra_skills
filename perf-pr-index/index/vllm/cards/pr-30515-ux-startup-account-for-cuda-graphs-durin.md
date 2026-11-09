# PR #30515: [UX][Startup] Account for CUDA graphs during memory profiling

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-03-07 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | ready, v1, nvidia, ready-run-all-tests |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/30515 |

## 一句话结论

启动内存剖析时用小型 KV cache 实捕 CUDA graph 估算其显存占用并从 KV cache 预算中扣除,避免 KV cache 过大导致运行期 OOM。

## 动机

启动期 memory profiling 并未覆盖所有 GPU 内存消耗方,用户只能靠 `--gpu-memory-utilization` 手动兜底。最显著的遗漏是 CUDA graph:它依赖 KV cache,只能在 profiling 之后捕获,profiling 时无从计入。结果是 KV cache 被分得过大,后续 graph 捕获阶段可能触发 OOM。本 PR 通过在 profiling 期间做一次"哑捕获"来估算 graph 显存,把这个量纳入预算。

## 优化点

新增 `profile_cudagraph_memory()`:先用 `max_cudagraph_capture_size`(或 1)个块初始化一个极小 KV cache,在一个独立的临时 graph pool(避免污染主 pool 造成碎片)中对每个 CUDAGraphMode 取前两个 BatchDescriptor 真实执行 warmup + capture,以 `torch.cuda.mem_get_info` 前后差值得到"首次捕获字节数"(共享内存,近似池底价)与"每图增量字节"(下限 1 MiB 驱动开销),再按 `(mode, 图数)` 外推:总估算 = max(各 mode 首捕) + sum(各 mode per-graph × (图数−1));FULL 与 PIECEWISE 运行时共享全局 pool 且不会并发重放,取 max 避免重叠部分被重复计数。`_dummy_run` 增加 `profile_seq_lens` 以按 `max_num_tokens // num_tokens` 估计 attention workspace 随上下文长度的开销;capture 结束后经 `CUDAGraphWrapper.clear_all_graphs()`(新增 WeakSet 实例登记)清空所有图、恢复原 pool、清理 profiling KV cache 并回滚 `num_cudagraph_captured` 计数。worker 侧在 `determine_available_memory()` 中把估算值同时计入 `peak_activation_memory` 并从 `available_kv_cache_memory_bytes` 中扣除,还会打印等价 `--gpu-memory-utilization` 换算提示。功能默认关闭,`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` 开启,并预告 v0.19 起转默认。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: profile_cudagraph_memory()/_warmup_and_capture()/最小 KV cache 构建与清理
vllm/v1/worker/gpu_worker.py: determine_available_memory 扣除估算值并输出 util 换算日志
vllm/compilation/cuda_graph.py: CUDAGraphWrapper 实例 WeakSet 与 clear_all_graphs()
vllm/v1/worker/gpu_ubatch_wrapper.py: graph_pool 改为代理属性,新增 clear_graphs
vllm/envs.py: 新增 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
```

估算核心(节选):

```python
first_capture = mem_samples[0]
per_graph = max(mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20)
shared_memory_estimate[mode] = first_capture
per_graph_estimate[mode] = per_graph * (len(descs) - 1)
...
total_estimate = max(shared_memory_estimate.values()) + sum(per_graph_estimate.values())
```

## 性能声明

无性能提升数字:PR 正文的 Test Result 为空,regex 亦未检出任何性能声明。收益是启动内存核算准确性(减少 OOM、允许用户更安全地调高 `--gpu-memory-utilization`);日志中会输出"actual vs estimated CUDA graph pool 用量及差值百分比"作为估算质量的持续校验,但正文未给出任何实测数字。PR 未注明测试条件。

## 验证与回归风险

Test Plan 给出的手测命令为 DeepSeek-R1 `-dp 8 --enable-expert-parallel` 加不同 `--gpu-memory-utilization`(结果未附)。回归风险面较宽:`_capture_cudagraphs` 的 warmup/capture 循环被重构进 `_warmup_and_capture`(每图后新增 synchronize),`cudagraph_dispatcher.get_capture_descs` 排序键改为 `(num_tokens, num_active_loras)` 降序,`gpu_ubatch_wrapper.graph_pool` 从实例字段变为代理属性;profiling 期间临时切换 graph pool 与清图逻辑若与 LoRA/EPLB/attention workspace 交互不当,可能影响正式捕获阶段(代码中以恢复 pool、`maybe_remove_all_loras`、`lock_workspace` 缓解)。默认关闭保证现网行为不变;开启后等效于降低 KV cache 预算,日志明示需要调高 `--gpu-memory-utilization` 以维持原有效 KV cache 大小。NVIDIA CUDA 平台专属(`torch.cuda.mem_get_info`)。

## 关联

同分类:本批次无其他 CUDA Graph 卡。功能上服务于 FULL/PIECEWISE cudagraph 捕获管线(`capture_model`、`VLLM_ENABLE_CUDAGRAPH_GC` 已有的 gc freeze 机制被抽为 `_freeze_gc` 复用);PR 讨论区(#30515#issuecomment-3739405494)确认默认关闭的决策。
