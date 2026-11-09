# PR #36720: [Bugfix][ROCm] Fix worker startup OOM on ROCm by skipping unreliable cudagraph memory profiling

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36720](https://github.com/vllm-project/vllm/pull/36720) |
| 作者 | JartX |
| 合并日期 | 2026-03-17 |
| labels | bug, nvidia, ready, rocm, v1 |
| 热度 | 👍 1 · 💬 22 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
ROCm 上 cudagraph 内存剖析依赖的 `mem_get_info`/`graph_pool_handle` 行为不同,产生虚高估计致 worker 启动即 OOM;在 ROCm 上跳过该剖析。

## 问题现象
ROCm/HIP 平台上 vLLM 无法启动,报错:

```
ValueError: Free memory on device cuda:0 (1.33/23.98 GiB) on startup is less
than desired GPU memory utilization (0.95, 22.79 GiB).
```

即启动时可用内存被判定为远低于实际值,worker 直接失败。

## 根因
正文披露:由 PR #30515([UX][Startup] Account for CUDA graphs during memory profiling)引入——它移除了 `profile_cudagraph_memory()` 与 `torch_peak_increase`、`non_kv_cache_memory` 的显式重算。`profile_cudagraph_memory()` 使用 `torch.cuda.mem_get_info()` 与 `graph_pool_handle()`,二者在 ROCm/HIP 上行为不同,可能产生不正确或负的内存估计;该估计被应用到 `available_kv_cache_memory_bytes` 后结果被抬高到超过实际空闲显存,导致 `request_memory()` 失败。变通环境变量 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` 之所以「碰巧能用」,是因为它强制应用了(错误的)估计,意外补偿了缺失的 `torch_peak_increase` 重算。

## 修复方案
改 `vllm/v1/worker/gpu_worker.py` 的 `determine_available_memory`:ROCm 平台跳过 cudagraph 内存剖析:

```python
cudagraph_memory_estimate = 0
# Skip on ROCm/HIP as graph pool handles and mem_get_info behave
# differently and can produce incorrect/negative estimates.
if not self.model_config.enforce_eager and not current_platform.is_rocm():
    cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()
```

ROCm 上估计恒为 0,后续 `cudagraph_memory_estimate_applied` 合并是 no-op;CUDA 上仍按 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 的设计生效(注释说明)。

## 影响与回归风险
- 影响窗口:受 #30515 之后到本修复之前的 ROCm 提交范围影响(正文指出引入 PR 为 #30515)。
- 回归面:仅改动 `vllm/v1/worker/gpu_worker.py` 的内存剖析逻辑。ROCm 上不再做 cudagraph 内存剖析,可用 KV cache 内存的估计不再包含 cudagraph 项(需靠运行时实际分配兜底);CUDA 路径行为不变,现有测试通过。正文验证:ROCm worker 启动无 OOM;CUDA 无行为变化。

## 关联
- #30515(引入问题的 PR:[UX][Startup] Account for CUDA graphs during memory profiling)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
