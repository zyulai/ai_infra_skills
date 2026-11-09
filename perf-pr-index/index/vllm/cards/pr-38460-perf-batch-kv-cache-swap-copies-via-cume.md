# PR #38460: [Perf] Batch KV cache swap copies via cuMemcpyBatchAsync

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Etelis |
| 合并 | 2026-04-03 |
| 分类 | KV Cache / Memory |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 csrc(cache.h/cache_kernels.cu/torch_bindings.cpp)、`vllm/_custom_ops.py`、`vllm/v1/kv_offload/worker/cpu_gpu.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/38460 |

## 一句话结论

把 KV cache CPU offloading 的逐 layer 逐 block 拷贝合并为一次 `cuMemcpyBatchAsync` 提交,handler 层提速 3.6-7.4x,Qwen2.5-7B 服务吞吐 +18%。

## 动机

KV offloading handler(`CpuGpuOffloadingHandlers.transfer_async`)原先对每个 tensor 调一次 `swap_blocks`,由 kernel 内部再按 block_mapping 循环逐块拷贝;块很小(4-32 KB)时,每拷贝的提交/launch 开销主导耗时,转移带宽远低于硬件上限。把所有 (src,dst,size) 对扁平化后用一次 driver 调用提交,可摊薄这笔开销。

## 优化点

新增 `swap_blocks_batch(src_ptrs, dst_ptrs, sizes)` op:接受 int64 CPU 张量形式的原始指针数组。CUDA 12.8+ 走 `cuMemcpyBatchAsync`,单次 driver 调用提交全部拷贝(`CU_MEMCPY_SRC_ACCESS_ORDER_STREAM`,直接 reinterpret_cast int64 指针避免拷贝);CUDA <12.8 与 ROCm 回退为 `cudaMemcpyAsync` + `cudaMemcpyDefault` 平坦循环。Python 侧 `CpuGpuOffloadingHandler` 在构造时预计算各 tensor 基址与块大小,`transfer_async` 里把所有 tensor × 所有 block pair 展开为平坦指针数组,替换掉原来的 per-tensor `swap_blocks` 循环。零额外 GPU 显存,行为不变;Supersedes #38216。

## 关键改动

```text
csrc/cache_kernels.cu: 实现 swap_blocks_batch(CUDA 12.8+ 批量拷贝/回退循环)
csrc/cache.h + csrc/torch_bindings.cpp: 注册 _C_cache_ops.swap_blocks_batch
vllm/_custom_ops.py: swap_blocks_batch Python 包装
vllm/v1/kv_offload/worker/cpu_gpu.py: transfer_async 改为构建平坦指针数组并一次提交
```

```cpp
+  CUmemcpyAttributes attr = {};
+  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
+  size_t attrs_idx = 0;
+  size_t fail_idx = 0;
+  CUresult result = cuMemcpyBatchAsync(
+      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(dst_data)),
+      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(src_data)),
+      reinterpret_cast<size_t*>(const_cast<int64_t*>(size_data)),
+      static_cast<size_t>(n), &attr, &attrs_idx, 1, &fail_idx,
+      static_cast<CUstream>(stream));
```

## 性能声明

8x H100 80GB,CUDA 12.8/12.9,baseline 为 `vllm==0.18.0`,每模式独立进程:

- Handler 级(BF16、FlashAttn KV 形状、5 warmup + 100 次):LLaMA-8B 64 blocks 16358 → 4519 us(**3.6x**);LLaMA-70B TP=4 20427 → 3299 us(**6.2x**);Qwen2.5-0.5B 128 blocks 23484 → 3184 us(**7.4x**);其余配置 4.0x-6.2x。
- E2E KV 转移带宽(OffloadingConnector kv_both,CUDA events 计时):Qwen2.5-1.5B 25.8 → 31.8 GB/s(**+23%**)、Qwen2.5-7B 34.2 → 40.7 GB/s(**+19%**)、Phi-3.5-mini +3%(51.6 → 53.3 GB/s),其余 +7%~+8%。
- 服务吞吐(8 线程持续负载):Qwen2.5-7B 6.0 → 7.1 req/s(**+18%**),p99 TTFT 6077 ms → 117 ms;LLaMA-3.1-8B 与 Mistral-7B 吞吐持平。作者归因:Qwen2.5-7B 只有 4 个 KV head,单块拷贝更小、提交开销占比更高。

## 验证与回归风险

PR 未声明新增单测;验证为上述 handler 级与 E2E 三层基准。风险面:改动集中在 KV offloading 路径(OffloadingConnector),不影响常规推理;CUDA <12.8 与 ROCm 走回退分支,收益依赖 CUDA 版本;指针数组在 CPU 构建、长度为 tensor 数 × block pair 数,超大转移规模时 host 端构建成本上升(PR 未量化);`cudaMemcpyDefault` 依赖 driver 从指针推断方向,源/目的必须按 VM 属性可区分(PR 未展开讨论)。

## 关联

同 KV offloading 区域:#61019(OffloadingConnector partial tails 与 EAGLE/MTP)。Supersedes #38216。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
