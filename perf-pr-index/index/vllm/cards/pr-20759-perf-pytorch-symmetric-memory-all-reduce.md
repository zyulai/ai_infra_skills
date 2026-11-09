# PR #20759: [PERF] PyTorch Symmetric Memory All-Reduce

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-08-22 |
| 分类 | Parallelism / Communication |
| Label | documentation, performance, ready |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/20759 |

## 一句话结论

新增 PyTorch symmetric memory TP all-reduce 路径(`VLLM_ALLREDUCE_USE_SYMM_MEM=1`),中尺寸输入 TTFT 提升最高 8-10%。

## 动机

custom allreduce 的 two-shot 实现在中等偏大的输入上性能劣于 NCCL,而 NCCL 在小输入上又不如 custom allreduce。PR 的思路是:按 world size 与 SM 架构给 custom allreduce 设输入大小上限(超过即不走它),同时引入基于 `torch.distributed._symmetric_memory` 的第三条 all-reduce 路径填补中等尺寸区间的性能空档。

## 优化点

新增 `SymmMemCommunicator`:启动时按 `SYMM_MEM_ALL_REDUCE_MAX_SIZES`(按 SM 9.0/10.0 与 world size 查表)预分配一块 symmetric memory buffer 并 `rendezvous`,校验 multicast 指针可用。`should_use_symm_mem` 仅接受 bf16、字节数被 4 整除且小于 max_size 的输入。all-reduce 时把输入拷入 symm buffer:world size 2/4 用 `two_shot_all_reduce_`,6/8(Hopper 下含 4)用 `multimem_all_reduce_`(组播硬件路径),完成后拷回。`CudaCommunicator.all_reduce` 的分派顺序变为 custom allreduce → symm mem → pynccl;启用 symm mem 时 `CustomAllreduce` 的 max_size 也按查表值收紧。

## 关键改动

```text
vllm/distributed/device_communicators/symm_mem.py: 新增 SymmMemCommunicator(buffer 预分配 + multimem/two-shot)
vllm/distributed/device_communicators/cuda_communicator.py: all_reduce 分派链插入 symm_mem_comm
vllm/distributed/device_communicators/all_reduce_utils.py: CUSTOM/SYMM_MEM_ALL_REDUCE_MAX_SIZES 尺寸表(按 SM 架构)
vllm/distributed/device_communicators/custom_all_reduce.py: symm mem 启用时收紧 custom allreduce max_size
vllm/envs.py: 新增 VLLM_ALLREDUCE_USE_SYMM_MEM 开关(默认关闭)
```

## 性能声明

Benchmark:`meta-llama/Llama-3.1-70B-Instruct`,`vllm bench serve` sonnet 数据集、60 秒、QPS 1/5/10,关闭 prefix caching。B200:TP=4 TTFT 最高提速 8%(如 QPS=1 mean TTFT 45.02→41.46 ms);TP=8 TTFT 提升 7-10%(QPS=1:42.84→37.01 ms)、TPOT 提升 5-7%。H100 TP=8:TTFT 最高提升 10%(QPS=10 mean 65.52→59.22 ms),TPOT 小幅提升。验证:GSM8K(5-shot)exact_match 0.9204(flexible-extract)。

## 验证与回归风险

新增 `tests/distributed/test_symm_mem_allreduce.py`(TP=2,与 `dist.all_reduce` 数值对比,atol 2.5/rtol 0.1)。风险较低:默认关闭,需显式设环境变量;仅支持 CUDA、bf16、SM 9.0/10.0 与有限 world size,其他条件自动回落原路径;`custom_all_reduce_utils.py` 重命名为 `all_reduce_utils.py` 属文件级改动,需注意引用同步(已同步更新)。

## 关联

同区域:custom allreduce(#2192 等)、QuickAllReduce(MI300)、PyNccl——本 PR 在同一 all_reduce 分派链上并列新增第三条路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
