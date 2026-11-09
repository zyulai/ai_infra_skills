# PR #31470: [NVIDIA] Support flashinfer Mega Moe

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenscarl(主要作者 @djns99,fork 自 djns99/mega_moe_flashinfer 分支) |
| 合并 | 2026-09-10 |
| 分类 | MoE (tier: major) |
| Label | documentation, high priority, quant, dependencies, deepseek, run-ci, bypass-fastfail, release-highlight |
| Diff | 素材未提供 diffstat;diff 含 6 个文件(含 672 行新增的 flashinfer_megamoe.py),另有 26 个文件被省略 |
| 链接 | https://github.com/sgl-project/sglang/pull/31470 |

## 一句话结论

新增 flashinfer_megamoe MoE backend(融合 EP all-to-all + 专家计算的 megakernel):大 prefill 吞吐 +11.9%,小 prefill deepgemm 变体 +18.6%。

## 动机

PR 正文的 Motivation/Modifications 为空模板,未提供成文动机。从素材可见:该 PR 是 djns99 分支的 fork,把 FlashInfer 的 `moe_ep.MoEEpMegaLayer`(fused EP all-to-all + expert-compute mega kernel)包装为模型无关的 MoE runner backend,接入标准 FusedMoE dispatch -> run_moe_core -> combine 流程;mega kernel 经 deep_gemm symmetric buffer 在内部完成 EP 通信,因此 dispatcher 与 combine 为纯 no-op。正文主体是多轮 MoE backend 对比 benchmark 与配置验证逻辑。

## 优化点

新增 `--moe-runner-backend flashinfer_megamoe` 与 `--moe-a2a-backend flashinfer_megamoe` 两种选择,以及 `--flashinfer-a2a-dispatch-type {auto,bf16,nvfp4,mxfp8}` 参数(auto 时按量化格式推导)。三种量化各配一个 megakernel 配置:Block-FP8 走 `DeepGemmMegaMoeConfig`,NVFP4 走 `Nvfp4CutedslMegaMoeConfig`,MXFP8 走 `Mxfp8CutedslMegaMoeConfig`,权重经 FlashInfer 的 preprocess 函数预变换后绑定。三个环境变量:`SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK`(对称 workspace 容量,未设置时从 cuda_graph_max_bs/chunked_prefill_size 推导)、`SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE`(跨 rank combine 线上格式:bf16 精确默认,mxfp8/nvfp4 以小精度代价减少 combine 流量)、`SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE`(kernel 内 FC2 top-k 归约,删除多 GB combine 暂存区,代价是 bf16 无序求和非确定性)。实现细节上:几何相同的 MegaMOE 层共享 workspace(FlashInfer 自带池按张量身份 key 导致不命中);CUDA graph capture 期间临时替换 deep_gemm 的 `pack_ue8m0_to_int` 以移除 `.all()` 断言的 device->host 同步(上游 DeepGEMM#414);capture 前显式 eager 构建所有 mega 层避免 capture 中阻塞同步。

## 关键改动

```text
python/sglang/srt/layers/moe/flashinfer_megamoe.py: 新文件,MoEEpMegaLayer 的 layer 构建/权重预处理/fused forward 封装
python/sglang/srt/arg_groups/moe_hook.py: megamoe 后端校验(架构/量化/SM100/dp==tp)与 dispatch-type 解析
python/sglang/srt/arg_groups/choices.py, exec_.py: 注册新 backend 选项与 flashinfer_a2a_dispatch_type 参数
python/sglang/srt/arg_groups/overrides.py: mxfp8 默认 runner 与 shared-experts fusion 调整覆盖 megamoe
python/sglang/srt/environ.py: 新增三个 SGLANG_FLASHINFER_MEGAMOE_* 环境变量
```

```python
mega = MoEEpMegaLayer(
    bootstrap=BootstrapConfig(world_size=world_size, rank=rank, device=...),
    fleet_params=FleetParams(num_experts=..., max_tokens_per_rank=..., token_hidden_size=...),
    weights=None,
    backend=MegaConfig(megakernel=megakernel_config, preprocess_weights=False,
                       transformed_weights=transformed_weights),
)
```

## 性能声明

均为 DeepSeek-V4-Flash-NVFP4、TP4/DP4/EP4(DP-attention,`--chunked-prefill-size` 按 dp_size 内部分摊)、单节点同硬件。早期对比(`flashinfer_trtllm_routed` 为基线,random 8in/1024out):c32 输出吞吐 876.88 → 674.21 tok/s(-23.1%)、c128 -10.1%、c1024 +0.7%(噪声内);结论为低/中并发 trtllm_routed 更优,饱和后吞吐中性。7/27/2026 对比(trtllm 为基线):Prefill 2048 tok/rank——deepgemm megamoe 输入吞吐 38,884 → 46,117 tok/s(**+18.60%**)、mean TTFT -16.59%,flashinfer_cutedsl_megamoe 各变体 +6.19%~+9.06%;Prefill 8192 tok/rank——**combine_nvfp4 最优:151,213 → 169,175 tok/s(+11.88%)、mean TTFT -12.50%**,deepgemm 仅 +2.08%;Decode(in 32/out 1024、10,240 req、并发 8192)各 backend 吞吐 ±2% 以内,combine_nvfp4 8,886 tok/s(+1.17%),deepgemm 以 -1.86% 吞吐换 P99 TPOT **-13.38%**。GSM8K(8-shot,1,316 题):trtllm 0.948,三个 flashinfer 变体 0.951-0.956,差异在二项噪声(~0.6 pt)内,无精度回退。7/29/2026 复测(2048 tok/rank,开 autotune):combine_nvfp4 53,150.67 tok/s,较未调优 +13.8%/mean TTFT -14.2%,对 deepgemm +2.5% 吞吐与 -4.1% mean TTFT,但 deepgemm 尾延迟更好(P95 -10.2%、P99 -11.8%)。注意事项:各指标取重复运行最优;decode 每 backend 仅单次运行,flashinfer 变体间 0.6% 的吞吐差异属暗示性排序;in-kernel FC2-reduce 变体因已知问题被排除在对比之外。

## 验证与回归风险

启动时强校验:仅 DeepseekV2/V3/V32/V4、Glm4Moe、NemotronH(Puzzle)、Qwen2Moe/Qwen3Moe 架构;仅 MXFP8/ModelOpt NVFP4/FP4-expert/hybrid NVFP4 量化(标准 FP8 MoE 不支持);要求 SM100、`dp_size == tp_size` 且 `--enable-dp-attention`;`SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE != bf16` 与 `IN_KERNEL_FC2_REDUCE=1` 组合启动即报错。回归风险:NVFP4 路径不支持 per-token activation scaling;`_capture_safe_ue8m0_pack` 是对 deep_gemm 上游函数的临时替换(依赖 DeepGEMM#414 解决);开启 in-kernel FC2-reduce 后输出为 bf16 无序求和、不可 bit 级复现;`--flashinfer-a2a-dispatch-type` 与 `SGLANG_MOE_NVFP4_DISPATCH` 互斥。pack 记录 CI:Base 通过、Extra 与 AMD ROCm 10 失败。

## 关联

Fork 自 djns99/sglang 的 `djns99/mega_moe_flashinfer` 分支(@djns99 为主要作者);引用上游 deepseek-ai/DeepGEMM issue #414(capture-safe UE8M0 打包)。无其他 PR 编号记录。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
