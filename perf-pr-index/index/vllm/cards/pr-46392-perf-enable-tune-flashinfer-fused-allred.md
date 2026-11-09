# PR #46392: [Perf] Enable + tune FlashInfer fused allreduce at world_size=16 on SM 10.3 (GB300)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @majunze2001 |
| 合并 | 2026-06-25 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ready, verified |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 3 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/46392 |

## 一句话结论

在 SM 10.3(GB300)上解锁 world_size=16 的 FlashInfer fused allreduce+RMSNorm,标定 64 MiB 阈值,TP=16 不再回退 AR+独立 RMSNorm。

## 动机

FlashInfer fused allreduce+RMSNorm 在两处被 world size 卡住:`PassConfig.flashinfer_max_size()` 的 `FI_SUPPORTED_WORLD_SIZES` 只有 [2,4,8];`FI_ALLREDUCE_FUSION_MAX_SIZE_MB[capability]` 表对 SM 10.3(GB300)没有 `16` 条目。任一缺失时 `flashinfer_max_size(16)` 返回 None,`AllReduceFusionPass` 与独立 `FlashInferAllReduce` communicator 都会自我禁用,TP=16 只能走普通 all-reduce + 单独 RMSNorm。PR #46001 让 DeepSeek-V4 的 TEP=16 可用(block-FP8 shared expert sharding)后,这条路径变得可达,本 PR 是对应的 perf 跟进——为 ws=16 做与 #37756 对 ws=2/4/8 同样的事。

## 优化点

三处小改:(1) `FI_SUPPORTED_WORLD_SIZES` 加入 16;(2) `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` 的 capability 100 与 103 两张表都补 `16: 64`(64 MiB,mnnvl multi-node),阈值来自实测:mnnvl 相对 pynccl(multi-node 回退)在 28-56 MiB 胜出(1.04-1.32x)、64 MiB 打平、以上落败,且 64 恰好匹配 ws=2/4 的既有约定;作者指出这是真融合阈值的下界(fused op 的 RMSNorm 是"免费"的);(3) benchmark 脚本配套:`FI_BACKENDS` 环境变量可在多节点 sweep 时只跑 mnnvl(trtllm 跨节点会 hang),设备改用 `LOCAL_RANK` 映射以支持多节点。

## 关键改动

```text
vllm/config/compilation.py: FI_SUPPORTED_WORLD_SIZES 加入 16
vllm/compilation/passes/fusion/allreduce_rms_fusion.py: cap 100/103 两表补 16:64 阈值
benchmarks/kernels/benchmark_fused_collective.py: FI_BACKENDS 选择 + LOCAL_RANK 设备映射
```

关键片段(compilation.py):

```python
FI_SUPPORTED_WORLD_SIZES = [2, 4, 8, 16]
```

## 性能声明

GB300 NVL(4 节点 × 4 GPU = 16 同一 NVLink 域)、bf16、hidden=7168(DeepSeek-V4-Pro)、CUDA-graph、100 trials,ms/op。独立 all-reduce(`benchmark_device_communicators.py`,定阈值的表):mnnvl vs pynccl 28MB 0.122/0.161(1.32x)、42MB 0.180/0.211(1.17x)、56MB 0.238/0.249(1.04x)、64MB 0.271/0.271(平)、70MB 起pynccl 反超。融合 AR+RMSNorm(`benchmark_fused_collective.py`,佐证):standard(compiled) vs mnnvl:7MB 0.164→0.033(5.0x)、28MB 0.386→0.125(3.1x)、56MB 0.700→0.254(2.8x)、112MB 1.278→0.510(2.5x)、224MB 2.419→1.017(2.4x);one-shot 标志对 mnnvl 无效(1shot ≡ 2shot)。注意事项:该环境 `nccl_allocator` 链接失败(pip wheel 的 libnccl.so.2 不在 linker path),NVLS/symmetric memory 被禁用,standard 基线偏慢、融合加速比被抬高,故 64 阈值以独立 AR 表(mnnvl vs pynccl,不依赖 symmetric memory)为准。

## 验证与回归风险

验证为上述两个 benchmark(与 #37756 相同方法)+ ruff/mypy lint。作者明确列出未覆盖项:端到端 compiled(非 eager)TEP=16 serving 且 fused pass 激活的场景未跑——#46001 的 gsm8k 功能验证是在 `--enforce-eager` 下做的,绕过了编译融合 pass,端到端验证留作后续。风险面:阈值表按 (capability, world_size) 精确匹配,仅影响 SM10.0/10.3 的 ws=16;`FI_BACKENDS`/`LOCAL_RANK` 改动只触及 benchmark 脚本。PR 未注明新增 pytest。

## 关联

前序:PR #37756(SM 10.3 all-reduce communicator tuning,ws=2/4/8,本 PR 同机制扩展)、PR #46001(TEP=16 functional,依赖前置)、PR #40970(benchmark harness 修复)。同区域:PR #48064(把 quant fusion 也开放给 mnnvl 后端)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
