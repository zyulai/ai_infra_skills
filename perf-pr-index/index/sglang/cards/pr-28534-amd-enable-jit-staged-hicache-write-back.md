# PR #28534: [AMD] Enable JIT staged HiCache write-back and fix CPU-index crash

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @AMD-yanfeiwang |
| 合并 | 2026-07-09 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | hicache, run-ci, jit-kernel, bypass-fastfail, run-ci-extra |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/28534 |

## 一句话结论

让 ROCm 走与 CUDA 相同的 page_first + kernel JIT 分级 write-back 路径,修复 CPU-index 崩溃,write-back 带宽提升 1.07x-2.85x。

## 动机

ROCm 上 `page_first` + `kernel` HiCache write-back 在首个 prefill 就崩溃(`RuntimeError: Destination indices must be a CUDA tensor`)。根因:`HiCacheController.start_writing()` 在 `kernel` io-backend + `page_first` 布局下把 `host_indices` 留在 CPU,假定会由"经 device 中转、接受 CPU 目的索引"的分级 JIT write-back kernel 消费;该 JIT 路径被 `_is_cuda` 门控,ROCm 上被禁用后回退到普通 `transfer_kv_all_layer_mla_lf_pf` C++ kernel,其 launcher 断言 `dst_indices.is_cuda()`,于是所有 TP scheduler rank 崩溃。#28473 用强制 `layer_first` 布局绕过,但代价是 AMD 永久失去分级 write-back 带宽,且 ROCm/CUDA 从此走不同代码路径。

## 优化点

从根因修复,使 ROCm 复用 CUDA 同一 JIT 路径。第一,kernel 层:`hicache.cuh` 把 NVIDIA 专属 PTX `ld/st.global.L1::no_allocate`(`load_nc`/`store_nc`)用 `#ifndef USE_ROCM` 包裹,为 ROCm 提供等价实现——基于 Clang `ext_vector_type(2/4)` 原生向量类型的 `__builtin_nontemporal_{load,store}`(配合 `__builtin_bit_cast`),确保生成确定性的向量化 `global_{load,store}_dwordx{2,4}` 非临时访存,而不依赖 LoadStoreVectorizer 合并。第二,设备匹配:`utils.cuh` 引入平台无关的 `kDLGPU`/`kDLGPUHost`(ROCm 下分别为 `kDLROCM`/`kDLROCMHost`),`hicache.cuh` 与 `staged_write_back.cuh` 的 TensorMatcher 改用它们,从而接受 ROCm 设备与 pinned host 张量。第三,Python 层:`memory_pool_host.py` 与 `pool_host/mha.py` 把 `can_use_jit`/`can_use_write_back_jit` 从仅 CUDA 扩到 HIP;`cache_controller.py` 改为仅在分级 JIT kernel 实际可用时才把 `host_indices` 留在 CPU,否则移到设备,保证任何无 JIT kernel 的后端上 `kernel` io-backend 都正确。

## 关键改动

```text
python/sglang/jit_kernel/csrc/kvcacheio/hicache.cuh: ROCm 非临时 load/store 实现
python/sglang/jit_kernel/csrc/kvcacheio/staged_write_back.cuh: matcher 接受 kDLROCM
python/sglang/jit_kernel/include/sgl_kernel/utils.cuh: kDLGPU/kDLGPUHost 平台抽象
python/sglang/srt/mem_cache/memory_pool.py 及 pool_host/mha.py:
  can_use_jit / can_use_write_back_jit 扩展到 HIP
python/sglang/srt/managers/cache_controller.py: 无 JIT 时 host_indices 移到设备
test/registered/jit/test_hicache_page_first_write_back.py: 新增单测(CUDA+AMD CI)
```

## 性能声明

write-back microbenchmark(复刻 #21631 设置):AMD MI355X / gfx950,ROCm 7.2,dtype=bf16,page_size=64,batch_pages=64,total_pages=128,`triton.testing.do_bench(warmup=5, rep=25)`,MHA 带宽计 K+V、MLA 计单 buffer。gfx950 上 `*_lf_pf` baseline 带宽约 15-30 GiB/s,分级 JIT 路径约 25-50 GiB/s: **MHA 1.21x-2.85x 加速,MLA 1.07x-1.86x 加速**(最小 1.07x 出现在 MLA num_layers=16/item_bytes=512:24.03 vs 25.68 GiB/s)。作为对比,CUDA 上 #21631 报告约为持平(0.87-1.25x),因 CUDA baseline 已接近峰值。ROCm 无 `cudaMemcpyBatchAsync` 等价物,`staged_write_back.cuh` 使用非 batch 回退(device relayout + 逐页异步 H2D 拷贝)。

## 验证与回归风险

新增 `test_hicache_page_first_write_back.py`,同时注册 CUDA CI(base-b-kernel-unit)与 AMD PR-CI(jit-kernel-unit),覆盖 MHA/MLA、element_dim 多组、含 staging 容量上下的 page 数(1/64/65/129)以触发单遍与多 chunk 分支,并显式验证 CPU-resident `host_indices` 的 write-back 与 prefix-cache-hit load 两条路径。功能验证:JIT HiCache 模块在 gfx942/gfx950(ROCm 7.2)以 hipcc 编译加载,gfx950 ISA 检查确认非临时指令带 `nt` 标志;2P1D Kimi-K2.6-MXFP4 分离部署(TP=8,page_first+kernel,write_through)AgentX v0.3 trace replay 并发 64,328 请求 0 错误。风险:ROCm 非临时访存路径是新写的(不依赖 LoadStoreVectorizer),`can_use_jit` 扩到 HIP 后 CUDA 上的行为不变但 ROCm cache_controller 分支逻辑变更;PR 建议落地后 revert #28473,否则其 fallback 会继续短路掉正确路径。

## 关联

前序:#28473(layer_first fallback,本 PR 要求将其 revert)、#21631(write-back microbenchmark 与 CUDA 侧分级 kernel 基线)。同区域:#27010(HiCache PP 一致性修复)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
