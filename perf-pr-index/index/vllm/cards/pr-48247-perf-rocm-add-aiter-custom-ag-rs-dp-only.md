# PR #48247: [Perf][ROCm] Add AITER custom AG/RS (DP only)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simondanielsson |
| 合并 | 2026-09-11 |
| 分类 | Attention / Kernel(major) |
| Label | performance, rocm, ready, nvidia |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(8 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/48247 |

## 一句话结论

ROCm 上把 AITER 自定义 all-gather/reduce-scatter 用于 DP 组(仅 uniform batch + FULL CUDA graph),DP attention 场景 TPOT 约 +3%。

## 动机

DP attention + TP experts 架构下,DP 组的 dispatch/combine 走 `all_gatherv` / `reduce_scatterv`,此前只能用 RCCL(Pynccl),通信开销影响 decode TPOT。AITER 的自定义 all-reduce 已在 ROCm 上接入,但其 AG/RS 能力尚未用于 DP 通信。属于 #48255 的一部分。

## 优化点

`CudaCommunicator` 在组名含 "dp"、world size ∈ {2,4,8}、ROCm 且 `VLLM_ROCM_USE_AITER_CUSTOM_AR` 开启(默认 True)时初始化 `AiterCustomAllreduce` 并置 `use_aiter_ag_rs`;`all_gatherv` / `reduce_scatterv` 在 `_can_use_aiter_ag_rs` 成立(sizes 一致、forward context 为 `CUDAGraphMode.FULL` 且 batch uniform)时优先走 AITER 自定义 kernel,失败自动回退 RCCL。TP/EP 组上的集合通信(TP all-reduce、EP a2a、sequence parallel a2a)完全不受影响;mixed batch 仍走 NCCL/RCCL。AITER AG 无 grouped 接口,hidden / topk_weights / topk_ids 分三次调用。`graph_capture` 修正为按各 group 自己的 comm 捕获(避免全局查找导致 double-capture),并把 DP group 纳入 graph capture 上下文;`aiter_ar_comm` 的 all-reduce 派发收紧到 `use_aiter_allreduce` 条件。可用 `VLLM_ROCM_USE_AITER_CUSTOM_AR=0` 关闭。

## 关键改动

```text
vllm/distributed/device_communicators/cuda_communicator.py: DP 组初始化 AITER AG/RS 与派发逻辑
vllm/distributed/device_communicators/aiter_custom_all_reduce.py: 暴露 should_custom_ag/rs 与 custom AG/RS 接口
vllm/distributed/parallel_state.py: graph_capture 捕获各 group 自己的 comm,纳入 DP group
benchmarks/kernels/benchmark_device_communicators.py: 新增 AG/RS 基准(按 collective 分表)
tests/distributed/test_rocm_aiter_custom_ar.py: DP2 eager 与 CUDA graph 下的 AG/RS 正确性测试
```

## 性能声明

8x MI300,`deepseek-ai/DeepSeek-V3-0324`(DPA + TP),对比 nightly `vllm/vllm-openai-rocm:nightly-2afa3f7`:总体 "~3% improved TPOT on 1k/1k (1-256 conc)"。1k/1k P50 TPOT:c1 +12.46%、c4 +12.50%、c8 +16.98%、c64 +2.19%、c128 +2.82%、c256 +3.49%,但 c16 -5.37%、c32 -2.57%;8k/1k 改善不明确且波动(c16 +9.28%,c2 -11.53%、c32 -7.02%)。kernel microbenchmark(world 2,bf16,hidden 8192):aiter_ag 比 pynccl_ag 快 1.13–1.16×,aiter_rs 快 1.29–1.30×。GSM8K 5-shot:flexible-extract 0.9469 vs nightly 0.9484(无非回归门禁声明,差异未讨论)。

## 验证与回归风险

验证:`tests/distributed/test_rocm_aiter_custom_ar.py::test_rocm_aiter_custom_ag_rs`(DP2,eager 与 CUDA graph capture 两种模式,与 `dist.all_gather_into_tensor` / `dist.reduce_scatter_tensor` 精确比对,2 passed);`test_aiter_all_gather_precedes_pynccl` 单测;bench serve sweep(1k/1k、8k/1k,10x prompts、2x warmups)。回归风险:仅 uniform batch + FULL CUDA graph 的 DP 组快路径生效,其他路径行为不变;world size 限 2/4/8;`reduce_scatterv` 中对称内存相关路径因 sizes 处理变化需留意;非 uniform sizes 或 mixed batch 走原 RCCL 路径,不影响正确性;高并发(16/32)出现 TPOT 回退,作者未解释。

## 关联

属于 #48255 的一部分;同区域为 ROCm AITER 集合通信(custom all-reduce 此前已存在);pack 正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
