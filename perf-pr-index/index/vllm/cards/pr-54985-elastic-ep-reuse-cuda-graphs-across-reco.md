# PR #54985: [Elastic EP] Reuse CUDA graphs across reconfiguration

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @itayalroy |
| 合并 | 2026-09-12 |
| 分类 | MoE（tier: major） |
| Label | ci/build, nvidia, mrv2 |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 因超长截断（14 个文件未展开），可见部分改动 `vllm/config/parallel.py`、`vllm/distributed/device_communicators/{all2all,base_device_communicator}.py`、`vllm/distributed/elastic_ep/{elastic_execute,elastic_state}.py`、`vllm/distributed/eplb/eplb_state.py`、`vllm/engine/arg_utils.py`、`vllm/envs.py`、`fused_moe/all2all_utils.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/54985 |

## 一句话结论

Elastic EP 重配置时复用 fused-MoE runtime 与 CUDA graphs（新 rank 在 preparation 期异步 warmup/capture），把缩扩容服务停顿从约 40s 降到 1–1.9s（降 95–97.5%）。

## 动机

异步 preparation（#47288）与 eager 模式优化（#51885）已把大部分 Elastic EP 步骤移出阻塞 commit 路径，但 CUDA graph 模式下每次重配置仍要在 commit 阶段做 kernel warmup 与 CUDA graph capture，期间服务被阻塞（现有请求不前进、新请求被拒）。目标：把这部分工作彻底移出 commit。

## 优化点

两条机制。其一，存量 rank 复用 fused-MoE runtime 与 CUDA graphs：NIXL EP 的 dispatch/combine buffer 在 ranks 连接/断开/mask/unmask 时保留 CUDA graph 条目（`NixlEPAll2AllManager` 的 buffer 按 `elastic_ep_max_dp_size` 预留最大容量，`max_num_ep_ranks` 改为从 buffer 状态读取；`physical_to_logical_map`/`expert_load_pass` 改为最大容量 buffer 上的视图，`reconfigure_physical_expert_slots` 只调整视图范围并填充新 slot），EPLB mapping 与 dispatch 输出 buffer 等图内张量保持同一内存地址，因此无需重捕获；`switch_and_prepare`/`commit_scale_up`/`commit_scale_down` 在可复用时跳过 `_release_cuda_graphs` 与 `warm_and_capture`。其二，新 rank 的 warmup 与 capture 移到 preparation 阶段：`warmup_local_kernels` 改为 `warmup_new_worker`（kernel warmup 后立即 warm_and_capture），期间存量 rank 继续服务——新 rank 的 dummy run 包在 `skip_dp_coordination`、`all2all_manager.mask_remote_ranks()`（mask 全部远端 peer）与 `_disable_flashinfer_autotune` 三个上下文中，FlashInfer 调优结果经 `sync_flashinfer_autotune_cache` 从存量 rank 拷贝。CUDA graph 复用当前仅对 NIXL EP 启用（`can_reuse_fused_moe_kernel` 判定 `all2all_backend == "nixl_ep"`），其他后端保留原重捕获路径。另新增 `--elastic-ep-max-dp-size` 参数（必须 ≥ 初始 data_parallel_size），移除 `VLLM_NIXL_EP_MAX_NUM_RANKS` 环境变量。

## 关键改动

```text
vllm/distributed/device_communicators/all2all.py: NIXL EP buffer 按最大容量初始化；stage_ep_size/commit_ep_size 拆分；mask_remote_ranks 上下文
vllm/distributed/elastic_ep/elastic_execute.py: 复用判定、新 rank preparation 期 warm_and_capture、autotune cache 同步
vllm/distributed/eplb/eplb_state.py: EPLB map/load 改最大容量 buffer + 视图；reconfigure_physical_expert_slots
vllm/config/parallel.py: 新增 elastic_ep_max_dp_size
```

```python
def can_reuse_fused_moe_kernel(parallel_config: ParallelConfig) -> bool:
    """The EP backend must preserve its CUDA graph entries when ranks are
    connected, disconnected, masked, or unmasked."""
    return parallel_config.all2all_backend == "nixl_ep"
```

## 性能声明

测试条件：全尺寸 `deepseek-ai/DeepSeek-V3` 原生 FP8，24 张 GB200 跨 6 节点，DP/EP 16→24→16 缩扩容、重流量下；vLLM 默认 model length/max seqs/CUDA graph capture；每个 revision 先跑一次预热 torch compile 缓存再在同一次分配上测量；两次重配置前后精度检查均通过。Downtime（现有请求不前进且新请求被拒的区间）：16→24 为 40.266s→1.019s（−97.5%）、24→16 为 38.320s→1.895s（−95.1%）。总重配置时间（含后台 preparation 与 commit）：16→24 为 79.571s→72.324s（−9.1%）、24→16 为 45.531s→9.406s（−79.3%）。附：max-capacity 输入对 expert kernel 的影响（active EP 固定 16，容量 64 vs 16）——Batched DeepGEMM decode +0.08%/prefill −0.97%；FlashInfer CuteDSL +1.21%/−0.89%；Batched Triton −17.58%/+13.08%；Batched CUTLASS FP8 −77.49%/−63.90%（padding 放大低效）。

## 验证与回归风险

测试：PR 正文报告缩扩容前后精度检查通过；diff 未展开测试文件（截断）。风险面：CUDA graph 复用仅限 NIXL EP 后端，其他 all2all 后端走原重捕获路径；DBO（UBatchWrapper）从直接 raise 改为 `clear_graphs()`；新 rank warmup 期间 mask 远端 rank、跳过 DP 协调，若与某些 backend 的隐式同步假设冲突可能死锁（PR 注释称 skip_dp_coordination 避免了原 dummy run 的 DP 死锁）；`elastic_ep_max_dp_size` 预留最大容量会增加 padding，Triton/CUTLASS batched expert 实现在扩容后吞吐受影响（表中 CUTLASS FP8 decode −77.49%）。

## 关联

前序：#47288（Elastic EP 异步 preparation）、#51885（eager 模式优化）。EPLB 相关：#23743/#28369（索引内 EPLB 卡片）。PR 正文 cc @tlrmchlsmth @SageMoore @ilmarkov，未列其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
