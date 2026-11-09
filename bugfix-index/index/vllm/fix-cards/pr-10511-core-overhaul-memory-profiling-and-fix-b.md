# PR #10511: [core] overhaul memory profiling and fix backward compatibility

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#10511](https://github.com/vllm-project/vllm/pull/10511) |
| 作者 | youkaichao |
| 合并日期 | 2024-12-16 |
| labels | ready |
| 热度 | 👍 11 · 💬 16 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
重构 GPU 显存 profiling:`gpu_memory_utilization` 改为单实例上限,修复多实例共享 GPU 时 KV cache 显存被少算的问题。

## 问题现象
正文未描述具体现象,仅说明本 PR 修复 issue #10451 并"明确内存分类与 profiling 流程"。改动前后 `--gpu-memory-utilization` 帮助文本语义反转:旧文本称其为"全局 GPU 显存利用率上限"(例:启动前已有 50% 显存被占用、设 0.9 则只有 40% 分配给模型执行器);新文本改为"per-instance 限制,只作用于当前 vLLM 实例,同卡跑两个实例可各设 0.5"。

## 根因
正文未披露详细根因,仅说明修复 #10451、补充初始 PyTorch 显存统计并与 PyTorch memory profiler 对齐。(由 diff 推断) 旧 `determine_num_available_blocks` 以 `total_gpu_memory * gpu_memory_utilization - peak_memory` 估算 KV cache 可用显存,其中非 torch 分配按 `total_allocated - torch_allocated` 的全 GPU 口径统计,把其他进程/其他 vLLM 实例占用的显存也计入了本实例开销,使 `gpu_memory_utilization` 实际表现为全局上限。新实现把单卡显存分为 3 类(本实例之外、本实例 torch、本实例非 torch),以实例初始化设备前的 baseline 为基准,只统计本实例的增量。

## 修复方案
- 显存 profiling 流程抽到 `vllm/utils.py` 的 `memory_profiling` 上下文管理器,新增 `MemorySnapshot`/`MemoryProfilingResult`,结束后计算 `torch_peak_increase_in_bytes`、`non_torch_increase_in_bytes` 与 `non_kv_cache_memory_in_bytes`;
- `vllm/worker/worker.py` 的 `determine_num_available_blocks` 改为在 `with memory_profiling(...)` 内执行 `profile_run`,按 `total_gpu_memory * gpu_memory_utilization - result.non_kv_cache_memory_in_bytes` 计算 KV cache 可用显存,并输出分段的新日志;
- `multi_step_model_runner.load_model` 同步透传 `model_memory_usage`;
- `vllm/engine/arg_utils.py` 更新 `--gpu-memory-utilization` 帮助文本为 per-instance 语义;
- 新增 `tests/entrypoints/llm/test_gpu_utilization.py`(同卡 3 个实例各设 0.3)与 `test_memory_profiling` 单测;调整 `tests/worker/test_profile.py` 期望值(torch 峰值 0.47 GiB、容差放宽到 100 block)。

## 影响与回归风险
- 回归面:所有 GPU worker 的显存 profiling 与 KV cache 块数计算路径(`determine_num_available_blocks`),KV cache 可用块数与旧版本可能不同;`multi_step_model_runner` 的 load_model;依赖旧 profiling 日志格式的解析工具;多条测试的期望显存值被相应调整。

## 关联
- https://github.com/vllm-project/vllm/issues/10451 (fixes)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
