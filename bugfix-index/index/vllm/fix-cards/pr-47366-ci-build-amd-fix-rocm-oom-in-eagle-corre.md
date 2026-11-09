# PR #47366: [CI/Build][AMD] Fix ROCm OOM in eagle_correctness_heavy by reserving CUDA graph memory

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47366](https://github.com/vllm-project/vllm/pull/47366) |
| 作者 | peizhang56 |
| 合并日期 | 2026-07-10 |
| labels | rocm, ready, v1, nvidia |
| 热度 | 👍 2 · 💬 11 |
| 分类 | Speculative Decoding |
| bug 类型 | resource |

## 一句话结论
AMD CI 上 TRITON_ATTN 的 eagle_correctness_heavy 用例在引擎初始化时 HSA_STATUS_ERROR_OUT_OF_RESOURCES(Free mem 0 MB);修复引擎间 VRAM 惰性回收(加 settle 屏障)与 ROCm 上 CUDA graph 内存未计入 profiling 两个独立问题。

## 问题现象
`test_eagle_correctness_heavy` 在一个进程里先后构建 reference 引擎与 speculative 引擎,参数化用例顺序执行。在 AMD "V1 e2e (4 GPUs)" 步骤上,TRITON_ATTN 的 Llama-4-Scout 用例(llama4_eagle、llama4_eagle_mm)在 CUDA graph capture 之后、引擎初始化时中止,报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`("Available Free mem : 0 MB");同一模型在 ROCM_AITER_FA 后端下通过。测试硬件为 AMD MI300 (gfx942)、4 GPU、TP=4、`VLLM_WORKER_MULTIPROC_METHOD=spawn`。修复后两个用例连续跑通过(2 passed in 2110.20s),屏障等待后各 GPU VRAM 降到约 0.28 GiB,无 HSA_STATUS_ERROR。

## 根因
PR 正文披露两个独立问题:

1. 引擎间 VRAM 惰性回收:ROCm 释放 VRAM 是惰性的,`cleanup_dist_env_and_memory()` 之后下一个引擎可能在上一引擎 VRAM 仍驻留时就开始 memory profiling,读到偏高的空闲内存,KV pool 因此被定得过大;
2. ROCm 上 CUDA graph capture 内存未保留(剩余 OOM 的根因):memory profiler 在 ROCm 上跳过了 CUDA graph profiling(#36720),KV cache 按 capture 零开销来定容,capture 只能挤进 `(1 - gpu_memory_utilization)` 的余量,当固定 capture 开销超过该余量时启动即 OOM。当时跳过是因为 `torch.cuda.mem_get_info()` 与 `graph_pool_handle()` 在 ROCm 上给出错误/负的估计;如今 #44825 已把 profiler 移到 `torch.accelerator.get_memory_info`(ROCm 上可靠),`graph_pool_handle` 也解析到 live capture 路径在 ROCm 上已使用的同一个 torch.cuda handle,两个前提都不复存在。

## 修复方案
1. `tests/v1/e2e/spec_decode/test_spec_decode.py` `_run_eagle_correctness`:在 ref 引擎与 spec 引擎 teardown 后各插入一次 `wait_for_rocm_memory_to_settle()`(复用既有 helper,非 ROCm 上为 no-op),让下一引擎在干净设备上 profiling;
2. `vllm/v1/worker/gpu_worker.py` `determine_available_memory`:CUDA graph 内存 profiling 的平台门控从 `current_platform.is_cuda()` 放宽为 `current_platform.is_cuda_alike()`(重新覆盖 ROCm;XPU 仍排除,见 #39977),并更新注释说明依据;
3. 移除早前针对单模型的 `gpu_memory_utilization=0.85` 覆盖(profiler 现在自动保留 capture 内存,按固定字节成本计,跨 MI300/MI325 等不同显存更可移植)。

关键 diff 摘录:

```diff
-                current_platform.is_cuda()
+                current_platform.is_cuda_alike()
                 and self.vllm_config.compilation_config.cudagraph_mode
                 != CUDAGraphMode.NONE
```

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围(ROCm CUDA graph profiling 跳过源自 #36720)。
- 回归面:改动集中在 `gpu_worker.py` 的内存 profiling 与 spec decode 测试文件。ROCm 上 `determine_available_memory` 现在会把 CUDA graph 估计计入可用内存,KV pool 相应变小,可能轻微降低 ROCm 下的最大 KV 容量换取启动不再 OOM;CUDA 路径行为不变(原本就启用),XPU 仍排除。测试文件改动仅影响 CI 用例执行节奏(settle 等待)。

## 关联
- #36720(ROCm 跳过 graph profiling 的来源)
- #44825(profiler 迁移到 torch.accelerator.get_memory_info)
- #39977(XPU 仍排除的依据)
- #46932(NVIDIA unified-memory 负估计修复,正文说明本 PR 与其不重复)
- @AndreasKaratzas 的 review(正文提及,覆盖移除 0.85 override 的讨论)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
