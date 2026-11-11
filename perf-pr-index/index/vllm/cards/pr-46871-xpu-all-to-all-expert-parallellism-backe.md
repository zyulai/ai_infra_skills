# PR #46871: [XPU] All-to-all Expert-Parallellism backend for batched MoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @oonyshch |
| 合并 | 2026-11-08 |
| 分类 | MoE(tier: major) |
| Label | performance, intel-gpu, verified |
| Diff | diffstat 缺失(pack 未提供 +/− 统计;pack 的 diff 另有 29 files 因篇幅被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/46871 |

## 一句话结论

为 XPU 新增 alltoall_batched MoE 专家并行通信后端与 BatchedTritonExperts 调优工具链(自述仍 WIP、e2e 尚有退化)。

## 动机

XPU 上 batched + routed expert parallelism 缺少专用 all-to-all backend。PR 自述:"Still WIP, e2e performance degradation, some bugs fixed. Depends on upcoming autotune."——即端到端性能此刻仍是退化状态,收益依赖后续的 autotune 机制;本 PR 打通的是机制与调优入口。

## 优化点

新增 `alltoall_batched` all-to-all backend:基于 `all_to_all_single`,把每个 token 只路由到拥有其 experts 的 ranks,激活以 batched format(E × capacity × K)组织。XPU 且非 monolithic 时用新的 `AllToAllBatchedPrepareAndFinalize`,其他平台回退 `make_moe_prepare_and_finalize_naive_dp_ep`。配套接线:`base_device_communicator.all_to_all_single()`(world_size=1 直接返回输入)、xpu_communicator 接受新 backend(仍借 AgRsAll2AllManager 取 group metadata)、`ParallelConfig`/`FusedMoEConfig` 的 `use_alltoall_batched_kernels`、activation_format 归入 BatchedExperts、`use_batched_dp_moe` 纳入。工具链:`benchmarks/kernels/benchmark_batched_moe.py` 以 per-expert capacity(`alltoall_batched` 下为 `max_num_batched_tokens × ep_size`)为 lookup key 调 `BatchedTritonExperts`,在 fill 1.0/0.1/0.01 三档(小 fill 模拟 decode 的稀疏占用)计时、按几何平均排名、每个候选先过 `torch_experts` 正确性检查,产出配置直接落入 vLLM configs 或 `VLLM_TUNED_CONFIG_FOLDER`;`merge_batched_moe_configs.py` 合并 per-capacity 配置(缺失 capacity fail-loud、单一 triton version guard)。另提交 Intel Arc Pro B60/B70(E=10,N=512)的 tuned configs。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/all2all_utils.py: use_alltoall_batched_kernels 时 XPU 选 AllToAllBatchedPrepareAndFinalize
vllm/config/parallel.py 与 fused_moe/config.py: 注册 alltoall_batched backend 与 batched 激活格式分派
benchmarks/kernels/benchmark_batched_moe.py 与 merge_batched_moe_configs.py(新增): 调优与配置合并工具
vllm/model_executor/layers/fused_moe/configs/E=10,N=512,...Arc(TM)_Pro_B60/B70...json(新增): tuned configs
tests/kernels/moe/test_alltoall_batched_capacity.py(新增): capacity 覆盖性测试
```

## 性能声明

PR 未注明 benchmark 结果;正文自述 e2e 性能仍为退化(still WIP),依赖 upcoming autotune,并称所有结果来自所述硬件的原始日志(附于 PR,pack 未包含)。另注:DeepSeek MLA on XPU 存在与本 backend 无关的 batched-prefill numerics bug(尚未 upstream),DeepSeek 运行需 `VLLM_MLA_DISABLE=1`。

## 验证与回归风险

新增 `tests/kernels/moe/test_alltoall_batched_capacity.py`:capacity 覆盖随机 routing(world 2/4/8,experts 40/64/256/16,topk 1–8)、全 token 指向同一 rank 时 token capacity 取等、hot expert 时 expert capacity 取等、activation buffer 不随 topk 缩放、slots 被 local expert 数 clamp、expert capacity 与 `BatchedTritonExperts.workspace_shapes()` 一致。风险:`activation_format` 分派新增 alltoall_batched 分支影响 MoE 内核选择;非 XPU 平台经 `make_moe_prepare_and_finalize_naive_dp_ep` 走 naive DP-EP 路径;backend 处于 WIP 状态。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
