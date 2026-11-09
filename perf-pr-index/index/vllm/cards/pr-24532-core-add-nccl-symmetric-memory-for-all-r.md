# PR #24532: [core] add nccl symmetric memory for all reduce

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Amir-19 |
| 合并 | 2025-09-23 |
| 分类 | Parallelism / Communication(tier: major) |
| Label | performance, ready, ci/build, v1 |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/24532 |

## 一句话结论

引入 NCCL symmetric memory allreduce 路径(ncclMemAlloc + ncclCommWindowRegister),TP8 大消息 allreduce 微基准最高约 2x,DeepSeek-R1 TP8 端到端输出吞吐 +3.6%。

## 动机

TP 推理中 allreduce 是关键集合通信。PyNccl 常规 allreduce 作用于普通 PyTorch 分配的缓冲,NCCL 无法将其识别为对称内存,因而用不上 NVLS(symmem)快速路径;而 CustomAllreduce/torch SymmMem 等自管方案各有规模/形状覆盖盲区。作者希望通过 NCCL 2.27.3+ 的 symmetric memory API(窗口注册)让 PyNccl allreduce 直接吃上 NCCL 原生优化。受 torch.compile 限制,当前需以 custom op 封装并把输入拷贝进 NCCL 注册的对称缓冲。

## 优化点

新增 `pynccl_allocator.py`:用 `load_inline` 编译 `ncclMemAlloc/ncclMemFree` 的 CUDAPluggableAllocator,挂到专用 `torch.cuda.MemPool`;`nccl_symm_mem_context` 进入该 pool 分配 symm 输入/输出缓冲,退出时对 pool snapshot 中新出现的 segment 调 `ncclCommWindowRegister` 注册为 NCCL 窗口(去重登记);CUDA graph 捕获期间先 `endAllocateToPool` 暂停 graph pool、退出后恢复,使 symm 分配与 graph capture 共存(graph_pool_id 由 `CUDAGraphWrapper`/ubatch 捕获线程注入)。新增 custom op `all_reduce_symmetric_with_copy`:symm pool 内分配输入/输出 → copy 输入 → PyNccl out-of-place allreduce(新版 `all_reduce` 支持 `out_tensor`),规避 torch.compile。分派在 `CudaCommunicator.all_reduce` 最前端,`should_nccl_symm_mem_allreduce` 按 world_size 与消息大小门控:min world_size 4,TP4 阈值 2 MiB、TP8 阈值 1 MiB,world_size > 8 恒启用;不满足则回落 quick-reduce/custom-allreduce/pynccl 原链路。开关为 `VLLM_USE_NCCL_SYMM_MEM=1`,需 NCCL ≥ 2.27.3、torch ≥ 2.8,运行时还需 `NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1`;NCCL 头文件缺失时编译失败自动降级禁用。`pynccl_wrapper.py` 增补 `ncclCommWindowRegister/Deregister` 与 `ncclGetRawVersion` 绑定。

## 关键改动

```text
vllm/distributed/device_communicators/pynccl_allocator.py: NCCL 内存池分配器、窗口注册、graph pool 暂停/恢复
vllm/distributed/device_communicators/pynccl.py: register_nccl_symmetric_ops 注册 custom op;all_reduce 支持 out_tensor
vllm/distributed/device_communicators/pynccl_wrapper.py: ncclCommWindowRegister/Deregister 绑定
vllm/distributed/device_communicators/all_reduce_utils.py: should_nccl_symm_mem_allreduce 门控
vllm/distributed/device_communicators/cuda_communicator.py: all_reduce 分派入口
tests/distributed/test_nccl_symm_mem_allreduce.py: 多进程正确性测试(并入 buildkite)
```

custom op 核心(节选):

```python
def all_reduce_symmetric_with_copy_impl(input_tensor):
    with nccl_symm_mem_context(pynccl_comm):
        symm_input = torch.empty_like(input_tensor)
        symm_output = torch.empty_like(input_tensor)
    symm_input.copy_(input_tensor)
    symm_output = pynccl_comm.all_reduce(symm_input, symm_output)
    return symm_output
```

## 性能声明

端到端(`benchmarks/benchmark_throughput.py`,DeepSeek-R1,TP=8,input 1000/output 1000,max-model-len 2000,dummy 权重,`--max-num-seqs 256`,1024 prompts):before 为 4.50 req/s、8989.34 total tok/s、4500.46 output tok/s;after(`VLLM_USE_NCCL_SYMM_MEM=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1`)为 4.66 req/s、9310.51 total tok/s、4657.81 output tok/s(输出吞吐 +3.6%)。硬件型号 PR 正文未注明。

kernel 微基准(benchmark_device_communicators.py,形状 (M, 8192),单位 ms,未注明显卡):TP8 下 8-64 MB 区间 pynccl-symm 最优,vs pynccl 1.53x-2.00x(8 MB 0.037 vs 0.074;64 MB 0.183 vs 0.281);1-4 MB 区间 symm_mem_multimem 最优(1.78x-2.22x);小消息 0.5 MB 为 ca_1stage 最优。TP4 下 pynccl-symm 在 2-47.84 MB 最优(1.34x-1.45x),0.5-1.5 MB 仍 ca_1stage 最优。TP2 场景 symm mem 不启用(world_size < 4)。

## 验证与回归风险

新增 `tests/distributed/test_nccl_symm_mem_allreduce.py`(world_size 2,与 dist.all_reduce 对照、atol 2.5/rtol 0.1),已加入 `.buildkite/test-pipeline.yaml`。回归风险:allreduce 分派最前端新增判断,所有 CUDA TP 用户受路径影响(默认关闭,VLLM_USE_NCCL_SYMM_MEM 门控);graph capture 期间 pool 暂停/恢复依赖 `_graph_pool_id` 正确注入(CUDAGraphWrapper 与 ubatch 捕获线程均已设置,否则 assert);版本硬约束(NCCL≥2.27.3、torch≥2.8.0a0)由 context 内断言/禁用兜底;编译失败仅告警降级。symm pool 额外占用显存(输入+输出双缓冲)未在 PR 中量化。

## 关联

同区域:torch symmetric memory 的 SymmMemCommunicator(multimem/two-shot,同一 benchmark 表对比项)、CustomAllreduce(索引卡 pr-2192)、PyTorch symmetric memory all-reduce 路径(索引卡 pr-20759-perf-pytorch-symmetric-memory-all-reduce)。本 PR 将 pynccl-symm 加入该对比矩阵并在 TP8 大消息区间胜出。
