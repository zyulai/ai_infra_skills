# PR #24111: [Kernels] Enable Torch Symmetric Memory All-Reduce By Default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-09-11 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含新增基准脚本约 486 行及 6 个文件修改 |
| 链接 | https://github.com/vllm-project/vllm/pull/24111 |

## 一句话结论

默认启用 torch symmetric memory TP all-reduce,按消息大小在 CA/symm mem/PyNccl 间调度,TP=8 E2E TTFT 最高提升 10%。

## 动机

torch symmetric memory all-reduce(multimem / two-shot)在 Hopper 与 Blackwell 上对中大型消息明显快于 PyNccl 与 custom allreduce,但此前由 `VLLM_ALLREDUCE_USE_SYMM_MEM` 控制且默认关闭,用户无法自动按输入大小选到最优算法。本 PR 将其默认打开,使 all-reduce 调度完全按 input size 分派,拿到现有算法中的最优组合。

## 优化点

三点机制改动:一是把 `VLLM_ALLREDUCE_USE_SYMM_MEM` 默认值从 False 改为 True,并在 `cuda_communicator.py` 中先初始化 SymmMemCommunicator、再把 symm_mem_enabled 传给 CustomAllreduce;二是收紧 CustomAllreduce 的 max_size(启用 symm mem 时取 min),并把 Blackwell("10.0")TP=6/8 的 CA 上限从 2MB 降到 1MB,让超过阈值的中大消息自动落到 symm mem / PyNccl;三是给 CustomAllreduce 增加 `VLLM_CUSTOM_ALLREDUCE_ALGO` 环境变量强制 1stage/2stage、给 SymmMemCommunicator 增加 force_multimem 与 max_size_override 测试选项,配合新增基准脚本对比各算法。

## 关键改动

```text
vllm/envs.py: VLLM_ALLREDUCE_USE_SYMM_MEM 默认 False -> True
vllm/distributed/device_communicators/all_reduce_utils.py: "10.0"(Blackwell)TP6/8 CA max size 2MB -> 1MB
vllm/distributed/device_communicators/cuda_communicator.py: 先建 SymmMemCommunicator,symm_mem_enabled 传入 CustomAllreduce
vllm/distributed/device_communicators/custom_all_reduce.py: symm mem 启用时 max_size 取 min(CUSTOM_ALL_REDUCE_MAX_SIZES)
csrc/custom_all_reduce.cuh: VLLM_CUSTOM_ALLREDUCE_ALGO 强制 1stage/2stage
benchmarks/kernels/benchmark_device_communicators.py: 新增 CA/PyNccl/symm mem 多算法基准(约 486 行)
```

## 性能声明

E2E:引自原 PR #20759,TP=8 时 TTFT 最高提升 10%。Isolated 基准(bfloat16,hidden 8192,张量 0.5–64MB,CUDA graph 捕获,H100 与 B200):H100 TP=2 下 ca_1stage 全尺寸最优(1.05x–2.82x vs pynccl);H100 TP=8 下 1–64MB 由 symm_mem_multimem 最优(最高 2.08x@4MB,16MB 1.68x);B200 TP=8 下 0.5MB ca_1stage 最优(2.12x),1.5–64MB multimem 最优(1.20x–1.74x);超大消息(如 H100 TP=4 64MB)pynccl 仍最优。TP=4 时 1–32MB 区间 multimem 与 CA 2stage 接近,故 32MB 以下仍全用 CA。

## 验证与回归风险

新增 benchmark_device_communicators.py 需 torchrun 多卡运行;`VLLM_CUSTOM_ALLREDUCE_ALGO` 校验非法值会直接 throw。回归面:所有 TP>1 的 CUDA 部署默认行为改变(CA 覆盖范围收窄到 1–2MB),TP=6/8 在 Blackwell 上受影响最直接;world size 或消息尺寸处于阈值边界时调度路径切换需要验证。

## 关联

前序:PR #20759(原始 torch symm mem all-reduce 引入)。
