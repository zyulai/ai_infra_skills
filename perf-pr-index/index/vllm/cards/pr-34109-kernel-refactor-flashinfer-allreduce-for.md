# PR #34109: [Kernel] Refactor FlashInfer allreduce for mnnvl backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hjjq |
| 合并 | 2026-02-26 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 3 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/34109 |

## 一句话结论

把 FlashInfer allreduce 接入新的 backend 可选 API,默认启用针对多节点 NVLink 优化的 mnnvl 后端,并允许 flashinfer allreduce 在无 rmsnorm 融合时(如 MoE+TP)独立使用。

## 动机

FlashInfer 新增了 allreduce API(flashinfer #2130)和面向 multi-node NVLink 场景优化的 `mnnvl` backend(flashinfer #1213)。上游基准显示 mnnvl 相对 vLLM 当时使用的旧 `trtllm` backend 性能相同或更好(flashinfer 仓库 allreduce.py 的基准),且 flashinfer 0.6.2 起 `backend=auto` 时总是选择 mnnvl。此前 #33985 已把 allreduce fusion pass 切到新 API,但只用了 `trtllm` backend;vLLM 还缺少在无 rmsnorm 可融合时(例如 MoE+TP)单独使用 flashinfer allreduce 的通道。本 PR 补齐 backend 选择与 standalone 使用两条路径。

## 优化点

机制上分三块:(1) 新增 `FlashInferAllReduce` device communicator,内含 primary workspace(standalone allreduce 与非量化 ar+rms fusion 共用)和独立的 quant workspace(仅 trtllm backend 支持量化融合),backend 由环境变量 `VLLM_FLASHINFER_ALLREDUCE_BACKEND=[auto, mnnvl, trtllm]` 选择(默认 auto,mnnvl 优先)。(2) `CudaCommunicator.all_reduce` 的回退链改为 quick reduce → flashinfer → custom allreduce → pynccl,standalone 使用由 `VLLM_ALLREDUCE_USE_FLASHINFER=1`(默认 0)开启,且 workspace 与 fusion pass 复用(fusion pass 先创建则 standalone 直接复用)。(3) allreduce+rmsnorm fusion pass 的 workspace 初始化迁到 `initialize_fi_ar_workspace`/`initialize_fi_ar_quant_workspace`;量化融合 pattern(FP8/NVFP4)仅在 trtllm quant workspace 存在时注册,mnnvl 模式下 layout_code 传 None(mnnvl 不支持量化融合与 swizzled layout)。另把 kernel benchmark 脚本扩展到两个 backend 并新增 16/64 的小序列长度。

## 关键改动

```text
vllm/distributed/device_communicators/flashinfer_all_reduce.py: 新增 FlashInferAllReduce communicator 与 workspace 管理(get/initialize/destroy)
vllm/distributed/device_communicators/cuda_communicator.py: all_reduce 回退链插入 flashinfer 分支
vllm/compilation/passes/fusion/allreduce_rms_fusion.py: 双 workspace 初始化、quant pattern 条件注册、layout_code 按 backend 区分
benchmarks/kernels/benchmark_fused_collective.py / benchmark_device_communicators.py: 支持 trtllm/mnnvl 两 backend 对比
```

关键片段(all_reduce 回退链):

```python
if (fi_ar_comm is not None and not fi_ar_comm.disabled
        and fi_ar_comm.should_use_fi_ar(input_)):
    out = fi_ar_comm.all_reduce(input_)
    assert out is not None
    return out
```

## 性能声明

PR 正文给出端到端 benchmark(`vllm bench serve`,sonnet 数据集,60 秒,模型 meta-llama/Llama-3.1-70B-Instruct,硬件 4xGB200):

- TP4(fuse enabled,mnnvl vs trtllm):qps=1 TTFT 43.33 vs 45.53、TPOT 9.53 vs 9.65;qps=5 TTFT 56.41 vs 57.46、TPOT 10.97 vs 10.99;qps=10 TTFT 68.13 vs 68.56、TPOT 13.69 vs 13.72。作者结论 "No TPOT slowdown in single node"。
- TP8(每节点 4xGB200,共 2 节点;trtllm backend 因多节点报错缺席):qps=1 下 mnnvl TTFT 57.8 / TPOT 8.55,standalone 55.33 / 9.15,baseline(无融合、无 standalone)57.31 / 10.75;qps=10 下 mnnvl 88.32 / 17.24,standalone 84.07 / 17.66,baseline 84.38 / 19.45(TPOT 相对 baseline 改善约 11%,TTFT 略有回退)。

另有 gsm8k 正确性 eval(5-shot,flexible-extract 0.9356 ± 0.0068,strict-match 0.8901 ± 0.0086)。

## 验证与回归风险

依赖 flashinfer >= 0.6.2 的新 allreduce API;fusion pass 在 workspace 创建失败(无 NVSwitch/multicast 不可用的拓扑)时禁用并回退非融合 allreduce。回归风险:allreduce 回退链插入 flashinfer 后,单机小消息路径的优先级变化可能影响非 GB200 硬件的默认行为(standalone 由默认关闭的环境变量控制);mnnvl 不支持 FP8/FP4 量化融合,量化模型在 auto 下若选中 mnnvl 需依赖 quant workspace 缺失时 pattern 不注册的守卫逻辑;多节点 trtllm backend 存在已知报错。

## 关联

前置:#33985(allreduce fusion pass 切换到 flashinfer 新 API,仅 trtllm);上游:flashinfer #2130(新 API)、#1213(mnnvl backend)。索引内同区域卡片:#20691(SM100 FlashInfer fused allreduce 集成)、#21069(FlashInfer allreduce+rmsnorm+quant fusion)、#20759/#24111(torch symmetric memory all-reduce)、#2192(custom allreduce kernels)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
