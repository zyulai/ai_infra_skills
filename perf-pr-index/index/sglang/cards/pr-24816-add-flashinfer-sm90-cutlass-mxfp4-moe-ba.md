# PR #24816: Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-05-13 |
| 分类 | MoE(tier: major) |
| Label | deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 可用,另有 3 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/24816 |

## 一句话结论

把 FlashInfer SM90 cutlass mixed-input MXFP4 MoE(W4A16)接入 GPT-OSS 与 DeepSeek-V4:Hopper 上 prefill 段(M≥1024)比 Marlin 快 24–36%,decode 段维持 Marlin,纯 opt-in。

## 动机

SM100 上 `--moe-runner-backend flashinfer_mxfp4` 已自动走 trtllm-gen,但 SM90 用户仍只能默认 Marlin。FlashInfer PR #3084(把 TRT-LLM PR #12451 移植进 `cutlass_fused_moe`)为 Hopper 带来 mixed-input 4-bit 权重 × 16/8-bit 激活的 LDSM+LUT 权重加载管线、tactic 剪枝(跳过 `CtaShape128x128x128B + COOPERATIVE` 对 `has_w4afp8` 的寄存器溢出组合)与 scheduler 调优(`max_swizzle_size=2`、`raster_order=Heuristic`)。本 PR 把该路径作为 opt-in 后端接入 SGLang 的两个 MXFP4 入口,SM90 默认行为不变(Marlin 仍是默认)。

## 优化点

- GPT-OSS 路径(`mxfp4.py`):`Mxfp4MoEMethod` 新增 `_fi_kernel` discriminator("trtllm_sm100"/"cutlass_sm90"/None);`create_weights` 按 128 对齐记录 padded intermediate/hidden(K % 128 约束,scale interleave factor 128//group_size=4)但不在加载 buffer 上 pad,保持 weight loader naive-copy 快路径正确;`_process_weights_for_sm90_cutlass` 把 HF 交错的 `[g,u,g,u,…]` 行 de-interleave 成 `[up; gate]` halved 布局、pad、再调 FlashInfer 的 byte/scale interleave helper;`_apply_sm90_cutlass` pad 输入、一次调用 `flashinfer_cutlass_fused_moe(use_w4_group_scaling=True)` 完成 GEMM1+SwiGLU+GEMM2(带 per-expert SwiGLU scalars α=1.702/β=1.0/limit=7.0 与 bias),输出裁回原宽度。
- DeepSeek-V4 路径(新文件 `mxfp4_flashinfer_cutlass_moe.py`):`Mxfp4FlashinferCutlassMoEMethod` 作为 Marlin/trtllm-gen 的 sibling;`Fp8MoEConfig.get_quant_method` 在 SM90 + `is_fp4_experts=True` 时运行时路由(SM100 继续 trtllm-gen)。权重准备:先走 FP8 base hook,`reorder_w1w3_to_w3w1`(cutlass 期望 `[up; gate]`),fp32 数值 E8M0 scale 经 `float8_e8m0fnu` cast 取 raw uint8 字节,再 byte/scale interleave;apply 时 bias=None(DSv4 无 MoE expert bias)、alpha=1/beta=0 标准 SwiGLU + config 的 `swiglu_limit`;并加入 `maybe_fuse_routed_scale_and_shared_add` 的 routed-scale + shared-add fusion。DSv4 标准形状(hidden=7168, inter=2048)已 128 对齐,只 assert 不 pad。
- `BypassedTopKOutput` 新增 `to_standard()`:cutlass kernel 需要显式 topk_ids/topk_weights,而 bypassed 格式下路由在 kernel 内部,需要时物化 routing 张量。
- 权重准备 helper 在 load 时只跑一次;FlashInfer < 0.6.11 时 import 被 try/except gate,用户选择该后端会得到明确报错。

## 关键改动

```text
python/sglang/srt/layers/quantization/mxfp4.py: GPT-OSS SM90 cutlass 分支(create_weights/process_weights/apply,~140 行)
python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py: 新文件,DSv4 Mxfp4FlashinferCutlassMoEMethod(~230 行)
python/sglang/srt/layers/quantization/fp8.py: get_quant_method 按 SM90/SM100 分发(~15 行)
python/sglang/srt/layers/moe/topk.py: BypassedTopKOutput.to_standard() 物化路由
python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py: fusion isinstance 检查纳入新类
```

## 性能声明

Kernel 级基准(`python/sglang/test/bench_mxfp4_sm90_kernels.py`):H100 80GB HBM3(SM 9.0),GPT-OSS-like 形状(hidden=4096, inter=2048, E=256, top_k=6),同一组随机 MXFP4 权重,FlashInfer `autotune(True)`,bias on/off 分别 autotune,每行 5 次 warmup 后取 30 次迭代中位数:

| tokens | Marlin (ms) | FI cutlass (ms) | 胜者 |
|---|---|---|---|
| 4 | 0.174 | 0.195 | Marlin +12% |
| 16 | 0.484 | 0.549 | Marlin +13% |
| 64 | 1.046 | 1.204 | Marlin +15% |
| 256 | 1.549 | 1.583 | 平局(±2%) |
| 1024 | 2.084 | 1.943 | FlashInfer +7% |
| 2048 | 3.420 | 2.663 | FlashInfer +28% |
| 4096 | 7.011 | 5.149 | FlashInfer +36% |
| 8192 | 12.541 | 10.085 | FlashInfer +24% |

bias 开销 ±2% 内。PR 明确:FlashInfer #3084 报告的 2.66–3.11× speedup 是对其自身旧 SM90 路径,不是 vs Marlin。结论:decode(M≤64)保持 Marlin 默认,prefill/chunked-prefill(M≥1024)与 PD 分离的 prefill worker 用本后端收益 24–36%。

端到端精度(chat-API GSM8K 200 题,T=0,H100):GPT-OSS-20B FlashInfer cutlass 0.940(triton_kernel baseline 0.945;latency 20.25s vs 17.44s,throughput 3338 vs 3926 tok/s),在公开 GSM8K 区间(~91–94%)内;DeepSeek-V4-Flash 0.985,总延迟 49.21s,输出吞吐 469 tok/s,与其公开区间一致。

## 验证与回归风险

新增 `python/sglang/test/test_mxfp4_sm90_cutlass.py` 12 例在 H100 全部通过,均为 bit-exact 对照:GPT-OSS 权重变换 5 例(含 2880×2880 生产形状与 192×192 pad 边界)、apply 对照 4 例、DSv4 apply 对照 3 例(验证 reorder + fp32→uint8 E8M0 cast + interleave 的 plumbing)。风险:要求 FlashInfer ≥ 0.6.11(旧版本 import gate 后明确报错);128 padding 带来约 2% MoE 权重内存开销(如 GPT-OSS hidden 2880→2944);SwiGLU scalars 硬编码 GPT-OSS 默认,其他模型接入需从 `moe_runner_config.gemm1_alpha`/`.gemm1_clamp_limit` plumb;autotuner 不按 bias 有无区分 tactic,切换 bias 需重跑 autotune(仅影响 cache hit-rate);DSv4 路径对非 128 对齐形状直接 raise。

## 关联

上游:FlashInfer PR #3084、TensorRT-LLM PR #12451;SGLang 前置依赖:PR #24452。同区域(MXFP4 MoE 后端):Mxfp4MarlinMoEMethod(SM90 默认)、Mxfp4FlashinferTrtllmMoEMethod(SM100)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
