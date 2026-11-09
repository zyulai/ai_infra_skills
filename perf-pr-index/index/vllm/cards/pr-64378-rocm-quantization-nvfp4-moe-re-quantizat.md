# PR #64378: [ROCm][Quantization] NVFP4 MoE re-quantization and fused shared experts through online quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-10-29 |
| 分类 | MoE(tier: major) |
| Label | performance, rocm, speculative-decoding, ci/build, multi-modality, tool-calling, qwen, deepseek, nvidia, quantization, mrv2, dflash, kimi, k3, minimax, DSv4.1 |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);pack 中 diff 仅保留测试类文件,其余 20 个源码文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/64378 |

## 一句话结论

让 ROCm 在没有 NVFP4 MoE backend 的平台上也能服务 NVFP4-expert checkpoint:routed experts 加载时在线 re-quantize 为 MXFP4/MXFP8,并让 fused shared experts(FSE)支持 online quantization。

## 动机

main 此前已支持把序列化存储的 NVFP4 MoE checkpoint(ModelOpt、compressed-tensors、Quark)re-quantize 为 online MoE 方法,但前提是平台存在 NVFP4 MoE backend;ROCm 没有该 backend,导致 MiniMax-M3.1-Flash 这类 NVFP4-expert checkpoint 无法在 AMD GPU 上服务。同时旧的 `RoutedExperts._fused_shared_expert_quantizer` 只处理全精度 shared expert,FSE 无法与 online quantization 组合,shared expert 只能走非融合的独立路径。

## 优化点

改动全部发生在加载期,前向路径不变。其一,NVFP4 oracle 在 `requantization_source_scope` 下跳过 backend 选择:checkpoint 量化方法只充当 dequant 源,`select_nvfp4_moe_backend` 在该 scope 内直接返回 emulation backend,从而不要求存在 NVFP4 kernel。其二,`QuantizeMethodBase.dequantize_weight` 对 MoE 改为按 expert 分组迭代返回 `(experts, w13, w2)`,以 chunk 为单位 dequant/re-quant,限制加载期 staging 内存峰值;ModelOpt、compressed-tensors、Quark 三个实现与唯一调用方一并迁移。其三,新增 `OnlineSharedExpertLoader`:加载时暂存 shared expert,加载完成后在线量化写入 routed expert 的槽位;FSE 兼容性检查改为比较 routed 与 shared expert 的有效 `(weight, activation)` quantization keys,不一致时拒绝 FSE 并告警,两种组合(BF16 routed 在线量化 + 量化 shared expert、带 FSE 的权重 reload)抛 `NotImplementedError`。另含两个 ROCm 小修复:gfx942 上 FP8 PTPC MoE 改用 torch 路径量化 W13 以规避 `dynamic_per_token_scaled_fp8_quant` 崩溃;MXFP8 Triton activation quant 在 M==1 时使用调优的 block size。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/oracle/nvfp4.py: NVFP4 oracle 在 requantization_source_scope 下返回 emulation backend;nvfp4_emulation_utils 按 _DEQUANT_CHUNK_BYTES 分块 dequantize_nvfp4_moe_weights
vllm/model_executor/layers/quantization/online/(mxfp4/mxfp8/requantization): online MoE 方法、requantization source 判定与 OnlineSharedExpertLoader
vllm/model_executor/layers/quantization/{modelopt,quark,...}: dequantize_weight 迁移为 MoE 迭代器,QuarkLinearMethod 支持 MXFP8
vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py: ROCm AITER FSE 与 shared expert 槽位路由
```

源码文件路径依据 PR 正文描述与测试 import 推断,diff 被截断未能直接核对。

## 性能声明

PR 正文未给出端到端性能数字。验证为单元测试:在 GB300(CUDA)上对本分支与基线 #64111 运行同一组 9 个量化/加载测试文件,本 PR 656 passed / 86 skipped / 27 failed;27 个失败与 #64111 完全相同且归因于环境(GPU 与其他任务共享约 10 GB 空闲、Triton "PTX compiled with an unsupported toolchain"、一次 OOM),相对基线的 10 个新增通过即本 PR 的新测试。

## 验证与回归风险

新增/依赖测试包括 `tests/quantization/test_online_shared_expert_fusion.py`(本 PR 主要覆盖:NVFP4 routed × online 目标 × FSE,compressed-tensors 与 Quark)、`tests/quantization/test_online.py`、`tests/kernels/quantization/test_nvfp4_emulation.py`(chunked dequant 与全量 dequant 逐位相等)、`tests/quantization/test_fp8_per_channel.py`(gfx942 避免 dynamic per-token 路径)等。风险点:核心覆盖测试为 ROCm-only 但其 CI 任务被禁用(依赖待锁定的 torchao),作者无 AMD 硬件、未实际运行,亦无 ROCm 端到端验证。影响路径:三种 NVFP4 MoE 实现的 `dequantize_weight` 签名变更、FSE 开启时的 shared expert 融合、layer-wise reload(带 FSE reload 抛 NotImplementedError)、gfx942 的 FP8 PTPC 加载。

## 关联

#63232(AMD 贡献者源分支,本 PR 是其拆分的第二个 PR)、#64111(stack 基底)、#63547(draft,在模型内部做 NVFP4→MXFP4 转换的替代方案,本 PR 走通用 online quantization API)、#61457(main 上的 per-layer FSE selection,本 PR 保留其 `resolve_layer_fused_shared_expert`)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
