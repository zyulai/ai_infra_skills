# PR #45876: [Perf] DFLASH: route draft bf16 GEMMs through tuned dispatch on GB300

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wookjeHan |
| 合并 | 2026-10-31 |
| 分类 | Speculative Decoding (tier: major) |
| Label | quant, jit-kernel |
| Diff | pack 未提供 diffstat（diff 内容可用：3 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/45876 |

## 一句话结论

DFLASH 草稿模型的小 M bf16 GEMM 接入 GB300 tuned dispatch 并补 split-K/warp split-K 表项，decode cycle 缩短约 1.1-1.4%（c=1）。

## 动机

低并发下 DFLASH 草稿模型每请求只有一个 `num_draft_tokens` 的 block，bf16 linears 跑在小 M 上，处于 cuBLAS 远低于显存带宽上限的区间。其中两条路径完全绕过 SGLang 的 tuned bf16 GEMM dispatch：`DFlashGroupedConv.kernel_projection` 是普通 `nn.Linear`；`_project_candidate_logits` 的 candidate head 直接 `torch.matmul`。其余多数 draft GEMM 形状在 GB300 上没有 tuned 表项，同样回退 cuBLAS。

## 优化点

三处接入/补表：1) `kernel_projection` 从 `nn.Linear` 改为 `ReplicatedLinear`，走 `UnquantizedLinearMethod.apply` 与 tuned dispatch；2) candidate head 在 backend 为 cutedsl/gemv、权重 bf16 且在 CUDA 上时改调 `bf16_gemm_dispatch`（kernel 按 GEMM 形状选择），否则保留 `torch.matmul`；3) 在 `unquant.py` 补 GB300 的 split-K / warp split-K 表项（M<=32，TP4 与 TP8），并在 `cutedsl_bf16_gemm.py` 为三个形状补 M=17..32 的 TGV M 范围。每个表项只有在 micro-benchmark 三次计时全部击败现有 kernel 才保留。表项按 (M, N, K) 键入，不专属 draft：hidden 8192 / intermediate 28672 / 64q8kv 的形状同样是 Llama-3-70B 级 dense 模型在 TP4/TP8 的形状。TP4 qkv 在 M=8/16 的新 warp split-K 表项取代原 GB300 TP16 split-K tactics（warp split-K 优先）。注意仅测了 GB300，未测 GB200。

## 关键改动

```text
python/sglang/srt/models/dflash.py: kernel_projection 改 ReplicatedLinear；candidate head 走 bf16_gemm_dispatch
python/sglang/srt/layers/quantization/unquant.py: GB300 split-K/warp split-K 表项（M<=32，TP4/TP8，共约 19 条）
python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py: 三形状的 TGV M 范围表项
```

## 性能声明

Micro-benchmark（GB300，CUDA graphs，cold L2，us/call，after-PDL）：kernel_projection (2048,8192) M=8 9.66->5.95（-38%）；TP4 gate_up M=16 -16%；TP4 qkv M=8 -15%、M=16 -10%（换 warp split-K）；TP4 candidate head (50512,8192) M=7 136.21->113.74（-16%）；TP8 qkv M=8 9.82->4.82（-51%）；TP8 candidate head M=7 -10%。E2E（Muse Spark Open NVFP4，GB300，DFLASH 8 drafts，两侧 FlashInfer autotune off，nsys mean decode cycle）：TP4 c=1 9354.9->9252.9 us（-1.09%）、c=2 -0.77%、c=4 -1.30%（含 ±1% MoE routing 波动，本 PR 改动的 GEMM 部分实际 -52.6 us/cycle 即 -0.40%）；TP8 c=1 7573.1->7465.7 us（-1.42%）、c=2 -1.24%、c=4 -0.52%。c>=5 无变化（draft GEMM M>32）。吞吐按 1/cycle 推算 c=1 约 +1.1%（TP4）/ +1.4%（TP8）。

## 验证与回归风险

准确度：Muse Spark Open（NVFP4, TP4, DFLASH）GSM8K 97.50%、AIME26 pass@1 95.83%±3.33%（16 runs）、pass@16 100%。`precompile_splitk_tactics()`（目前仅 Qwen4-Exp 调用）会编译所有表项，启动时多构建 19 个 tactics。风险：表项按形状全局生效，同形状的 Llama-3-70B 级 dense 模型在 SM10x + auto backend 下 M<=32 也会切到新 kernel；TP4 qkv 的 warp split-K 覆盖旧 TP16 split-K tactics；仅 GB300 实测。CI 状态在合并时显示为失败（Base/Extra/AMD 均 x），PR 正文未说明原因。

## 关联

DFLASH 同类 spec decoding 优化参见索引 Speculative Decoding 分类下其他条目（如 #34337 multi-adapter LoRA spec decoding）；PR 未直接引用前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
