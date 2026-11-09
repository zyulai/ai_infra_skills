# PR #17946: [MUSA][8/N] Port CUDA kernels that are compatible with MUSA

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yafengio |
| 合并 | 2026-04-24 |
| 分类 | Speculative Decoding |
| Label | quant, dependencies, sgl-kernel, run-ci, mthreads |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计);diff 有省略(13 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/17946 |

## 一句话结论

为摩尔线程 MUSA GPU 移植可兼容的 CUDA 内核,并新增 MUSA 专用 custom allreduce,打通 sgl-kernel 构建。

## 动机

这是 #16565 跟进的 Moore Threads GPU 全量支持系列之一:NVIDIA 之外的后端此前缺少 sgl-kernel 内核覆盖,LLM 推理在 MUSA 上跑不起来或只能走慢路径。目标是把编程模型兼容的 CUDA 内核移植为原生 MUSA 实现,同时保持 CUDA/ROCm/MUSA 代码库统一。

## 优化点

按 `USE_MUSA` 条件编译移植了一批 CUDA 内核,覆盖 custom AllReduce、elementwise(如 fused_add_rmsnorm)、GEMM(含 dsv3_fused_a_gemm/dsv3_router_gemm、bmm_fp8)、MoE 与 fused MoE(moe_align_block_size、topk_softmax、moe_sum_reduce、moe_fused_gate 等)、KV cache 传输工具(transfer_kv 系列)、speculative decoding(tree_speculative_sampling_target_only、verify_tree_greedy、build_tree_kernel_efficient 等)与 GGUF 量化。重点是 custom allreduce:MUSA 没有直接复用 CUDA 的 PTX release/acquire 内存语义,改用 MUSA 原生的 volatile_store/volatile_load、flushInv_byp 与 __syncthreads_lm 实现 barrier;并新增 MUSA 专用 `custom_all_reduce_2shot` 内核,用 warp shuffle(__shfl_xor_sync)做 rank 间 reduce、两段 barrier 组织 reduce-scatter + all-gather,替代 CUDA/NCCL 路径,为摩尔线程多卡推理打基础。allreduce 的线程/block 配置也按 MUSA 调整(1024 threads、60 blocks,CUDA 仍为 512/36)。构建侧新增 setup_musa.py 与 pyproject_musa.toml,并升级 torchada 依赖。

## 关键改动

```text
sgl-kernel/csrc/allreduce/custom_all_reduce.cuh: MUSA 原生 barrier 语义 + custom_all_reduce_2shot 内核
sgl-kernel/csrc/common_extension_musa.cc: 为 torch::kMUSA 注册 allreduce/elementwise/gemm/moe/spec 等 op
sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.mu: 新增 MUSA 版 fused add rmsnorm 内核
setup_musa.py / pyproject_musa.toml: MUSA 构建配置
```

## 性能声明

PR 正文无性能数字:"Benchmarking and Profiling" 一节为空,素材中也未检测到 perf claim。仅给出功能验证:在干净 torch_musa 容器中 `python setup_musa.py install` 构建安装 sgl-kernel 0.3.21 成功,并以 Qwen3-0.6B、tp2/dp2 + DP attention 拉起 server(sampling-backend flashinfer 暂时禁用)。

## 验证与回归风险

验证手段为 torch_musa 容器内的构建与 Qwen3-0.6B 双卡启动(见上)。回归面:所有改动都包在 `USE_MUSA` 条件编译或 `_musa` 后缀文件里,CUDA/ROCm 路径理论上不受影响,但 custom_all_reduce.cuh 中把 512/36 等常量参数化(kDefaultThreads 等)触及公共代码;REDUCE_CASE 宏被拆成 CUDA 与 MUSA 两份,后续改阈值时需两处同步。

## 关联

前序系列:#16565(MUSA 支持总追踪 issue/PR,标题为 [MUSA][8/N] 系列第 8 部分)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
