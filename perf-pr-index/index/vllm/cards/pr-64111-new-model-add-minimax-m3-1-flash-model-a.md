# PR #64111: [New Model] Add MiniMax-M3.1-Flash Model and Kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-10-29 |
| 分类 | Speculative Decoding \| tier: major |
| Label | performance, rocm, speculative-decoding, ready, ci/build, multi-modality, tool-calling, qwen, deepseek, nvidia, quantization, mrv2, dflash, kimi, k3, minimax, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注「139 file(s) dropped from diff for size (see PR for full diff)」 |
| 链接 | https://github.com/vllm-project/vllm/pull/64111 |

## 一句话结论

新增 MiniMax-M3.1-Flash 模型与全套 decode kernel、DSpark draft；正文未给定量加速比。

## 动机

本 PR 是 #63232 拆分后的第一个 PR，添加 `vllm/models/minimax_m3` 下的模型代码（NVIDIA 与 ROCm，含 DSpark draft）及全部 kernel 改动；#63232 中 model-runner、sampling、KV cache manager 的改动留待后续 PR。目的是在 Blackwell 与 ROCm 上支持 MiniMax-M3.1-Flash 新模型。

## 优化点

正文 Claims：Blackwell（SM100/103/107）上 MXFP8 checkpoint 的 NVFP4 routed experts 经带 per-token FC1 input scales 的 compressed-tensors NVFP4 MoE 加载，KV 与 index-K cache 以 NVFP4 跑在 MSA CUTLASS Q8KV4 kernels；DSpark speculative decoding 默认读取 checkpoint 的 `dspark/` 子目录，配 fused draft kernels 与用于 non-causal draft blocks 的 `FLASHINFER_TS` backend。Decode kernels 含 fused QKV+indexer MXFP8 GEMM epilogue、SM100 fused dense GEMM1、persistent QK-norm/RoPE/KV-insert kernel、small-batch MoE finalize 融进 all-reduce + Gemma norm + MXFP8 quant、DeepGEMM NVFP4 MegaMoE（opt-in 带 EP）、TP4 opt-in NVFP4 indexer decode CP。ROCm 侧含 gfx950 NVFP4 KV + NVFP4 indexer、one-launch-per-layer FlyDSL mono decode、small-M MXFP8 linear tuning、gfx942 fallbacks。正文强调共享代码改动均 opt-in 或对其他模型 no-op。

## 关键改动

素材 diff 仅保留 3 个文件，另有 139 file(s) dropped from diff；正文另点名模型目录、新 `FLASHINFER_TS` backend、`attention_config.indexer_decode_cp` 与共享原语目录。

```text
.buildkite/test_areas/kernels.yaml: 新增 B200 上 MegaMoE 与模型测试 CI 步骤
benchmarks/kernels/benchmark_mxfp8_tp4.py: 新增 MXFP8 TP4 linear / HIP graph replay 基准
csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu: 新增 fused QK-norm/RoPE/KV-insert kernel
vllm/models/minimax_m3/: 模型 + DSpark draft + kernels（正文点名）
vllm/models/common/ops/moe_tail_fusion: 共享 MoE tail CuTe DSL 原语（正文点名）
```

## 性能声明

PR 正文未给出定量性能数据（无加速比、吞吐或延迟的 before/after 对比）。正文给出精度与 DSpark 接受长度：TP4 GB300、NVFP4 KV + NVFP4 indexer、DSpark k=7 下，gsm8k（1319 题、5-shot、T=0）0.9287 flexible / 0.9196 strict，DSpark MAL 4.77；aime26 mean@16 0.9271（pass@16 0.9667），MAL 3.48（每 verify step 平均接受 token 数，满分 8）。#63232 全分支 aime26 plain 0.9458 / DSpark 0.9208、MMMU-Pro strict 0.7936 / 0.7861，paired 比较在噪声内。

## 验证与回归风险

单测在 GB300（SM103）上跑，MSA `25c1447` pinned、CUDA 13.4 用于 Q8KV4 JIT；`tests/kernels/attention -k minimax_m3` 219 passed / 64 skipped，另一组 689 passed / 5 skipped / 3 failed，`test_custom_all_reduce.py`（2 GPUs）57 passed / 5 skipped。3 个失败在 main 上同样失败，非本 PR 引起。正文明确 ROCm 未运行（无 AMD 硬件）、`VLLM_GPU_SYNC_CHECK=error` 未运行。可能受影响路径：共享代码均声明 opt-in / no-op；ROCm mono decode 与上游 #62712 MonoKernel 重叠需 reconcile；Q8KV4 prefill 的 per-device split workspace 仅 SM107 分配（权宜之计）。

## 关联

正文明确关联：#63232（被拆分的原 PR，本 PR 为第 1 部分）、后续 PR 2–7 拆分计划表（PR 3/5/7 依赖本 PR）、#62712（ROCm MonoKernel，与本 PR 的 `VLLM_ROCM_MINIMAX_M3_MONO` 重叠需 reconcile）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
