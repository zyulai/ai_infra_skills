# PR #34692: [ROCm] Enable DeepEP ROCm as all2allbackend for AMD GPUs.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lcskrishna(co-author @itej89) |
| 合并 | 2026-03-21 |
| 分类 | MoE(tier: major) |
| Label | documentation, rocm, ready, ci/build |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(7 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/34692 |

## 一句话结论

让 ROCm/DeepEP 成为 AMD GPU 上可用的 all2all backend:MI300 + CX7 集群可用 deepep_low_latency / deepep_high_throughput 对 DeepSeek-V3 做 expert-parallel 服务,并让 FusedBatchedMoE 支持 float8_e4m3fnuz。

## 动机

DeepEP 是 MoE expert-parallel 的 all-to-all 通信库,此前 vLLM 的 deepep all2all backend 只适配 NVIDIA 实现,AMD GPU 上不可用,MI300 集群部署 DeepSeek-V3 类 MoE 模型缺少 DeepEP 通信路径。本 PR 集成 ROCm/DeepEP 分支所需改动,使 AMD GPU 可开箱即用(OOB)地运行 DeepEP backend。

## 优化点

三方面适配。其一,`all2all.py` 的 normal/low-latency 两类 buffer 构造在 ROCm 下不传 ROCm DeepEP 尚不支持的 kwargs(`explicitly_destroy`、`allow_nvlink_for_low_latency_mode`、`allow_mnnvl`),留 TODO 待上游 API 更新后移除平台分支。其二,low-latency dispatch 在 ROCm 下不传 NVIDIA 专属参数(`round_scale`/`use_ue8m0`/`use_nvfp4`/`x_global_scale`)。其三,把 fused MoE 多处硬编码的 `torch.float8_e4m3fn` 统一替换为 `current_platform.fp8_dtype()`(MI300 为 `float8_e4m3fnuz`),并在 `fused_batched_moe.py` 的支持 dtype 列表加入 `float8_e4m3fnuz`、compute_type 判定改用平台 fp8 dtype,使 FusedBatchedMoE Triton kernel 能在 AMD 上以 fnuz 格式运行。另在 `Dockerfile.rocm_base` 加入 `liblzma-dev`、`pkg-config` 构建依赖。

## 关键改动

```text
vllm/distributed/device_communicators/all2all.py: ROCm 下裁剪 DeepEP buffer 构造 kwargs
vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py: ROCm low_latency_dispatch 参数分支
vllm/model_executor/layers/fused_moe/{config,utils,fused_moe,fused_batched_moe}.py: fp8 dtype 统一为 current_platform.fp8_dtype(),新增 float8_e4m3fnuz
docker/Dockerfile.rocm_base: DeepEP 安装依赖
```

## 性能声明

PR 正文未给出吞吐/延迟数字,仅给正确性结果。GSM8K(5-shot,lm_eval,DeepSeek-V3,fp8,MI300 + CX7,TP1 + EP):1N + deepep_low_latency exact_match flexible 0.9484 / strict 0.9462;1N + deepep_high_throughput 0.9484 / 0.9469;2N(DP16)+ Low Latency 0.9492 / 0.9492;2N + High Throughput 0.9492 / 0.9454;1P1D(NIXL/RIXL,UCX over CX7)0.9439 / 0.9447。

## 验证与回归风险

验证全部为手工 serve + lm_eval:1N(DP8)、2N(DP16,`--data-parallel-start-rank 8` headless)跑 `--all2all-backend deepep_low_latency | deepep_high_throughput`(LL 需 `ROCSHMEM_MAX_NUM_CONTEXTS=256`、`ROCSHMEM_HEAP_SIZE` 等环境变量),以及 1P1D NIXL KV transfer。PR 自述 "Unit tests to add"——单元测试尚未落地是主要验证缺口。回归风险:`fp8_dtype()` 替换在 NVIDIA 上应返回同一 dtype(e4m3fn)行为等价,但触及 fused MoE 量化路径多处;ROCm LL dispatch 分支不传 ue8m0/nvfp4 参数,仅限 ROCm;平台分支标注 TODO 待清理。

## 关联

#36086(Docker ROCm/DeepEP 安装步骤);上游依赖 ROCm/DeepEP 仓库。同分类 MoE 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
