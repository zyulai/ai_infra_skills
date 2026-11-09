# PR #58510: [ROCm][Perf] MXFP8 GEMM on native 32x32 block scales for gfx950

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Fangzhou-Ai |
| 合并 | 2026-09-25 |
| 分类 | Speculative Decoding \| tier: major |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 2 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/58510 |

## 一句话结论

gfx950 直接读 MXFP8 32×32 block scales 的 GEMM，MI355X kernel 中位 1.49x、端到端 TPOT −4.8%。

## 动机

DeepSeek V4/V4.1 checkpoints 以 32×32 blocks 存放 MXFP8 weight scales，但 ModelOpt loader 会把它扩展成每个 weight row 一条 scale，gfx950 上 `RocmDotScaledMxfp8LinearKernel` 因此在这些 per-row scales 上跑一个通用 `tl.dot_scaled` kernel。正文指出在 DeepSeek-V4.1-Flash 上这些 linear（`fused_wqa_wkv`、`wq_b`、indexer `wq_b`、`wo_b`、shared-expert `gate_up` 与 `down`）在 TP2 conc32 下占每 decode step GPU kernel time 的 11.7%。

## 优化点

保留 checkpoint 的 32×32 scale blocks，新增直接读取、按 V4.1-Flash 投影逐形状调优的 GEMM。新增 `rocm_block32_gemm.py`：针对 MXFP8 激活 × e4m3 权重、`[N/32, K/32]` E8M0 block scales 的 `tl.dot_scaled` GEMM，大 M 用 tiled kernel、小 M 用 packed kernel（用 K panels 而非 tokens 填满 MFMA 行、只保留 block-diagonal 乘积）；两者都可 split K，通常在同一 launch 内按 split 顺序归约（确定性、无需第二次 launch），若首次 split-K 落在 graph capture 内则回退到第二次 reduce launch。N/K 为 constexpr、M 为 runtime（`do_not_specialize`）避免按 batch 重编译，配置来自 MI355X 上为 TP2/TP4 十个投影调好的表，注册为 custom op。`RocmDotScaledMxfp8LinearKernel` 在每 32 行共享 scale 时恢复 compact blocks、`apply_weights` 按 scale 形状分派，原生支持 `K % 32 == 0`（此前 V4.1 shared-expert `down` TP4 的 K=576 在加载时被反量化成 BF16）；per-row (1×32) scales 的 checkpoint 仍走原 kernel。grouped `wo_a` GEMM（#58456）经同一 kernel、scale 也变 compact。

## 关键改动

```text
vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py: 新增读 32x32 block scales 的 tl.dot_scaled GEMM（tiled / packed / split-K）
vllm/model_executor/kernels/linear/mxfp8/rocm_native.py: RocmDotScaledMxfp8LinearKernel 恢复 compact scales 并按 scale 形状分派
tests/kernels/quantization/test_rocm_mxfp8_linear.py: 新增 block32 与 split-K 测试（MI355X 83 passed）
tests/kernels/attention/test_rocm_triton_attn_dsv4.py: 相关 attention 测试（MI355X 95 passed）
```

## 性能声明

Kernel（MI355X、gfx950）：经 vLLM entry point（custom op + table lookup），权重冷（移出 256 MB MALL）、HIP graphs、相同 MXFP8 输入、三个 kernel 同一 session。90 个点上 vs 当前 vLLM kernel 中位 1.49x（min 0.98x、max 2.37x），vs ROCm/aiter#5750（`0dbc6c98e4`，含 fused split-K 与 compact scales）中位 1.07x（min 0.95x、max 1.27x）；唯一低于 1x 的当前 kernel 点是 shared-expert `down` TP4 M=2048（14.8 vs 15.1 µs，对方为 BF16 hipBLASLt）。110 个 off-grid 点对 aiter 仍为 1.07x 中位、最低 0.94x。相对 fp32 reference 的最大相对误差 1.67e-3，与当前 kernel 相同。端到端：复现 InferenceX `dsv41flash_fp4_mi355x_vllm_mtp` job（TP2 conc32、DSpark x5、synthetic acceptance 3.51、`--max-num-batched-tokens 16384`、cudagraph capture 1024、3600 s 窗口、两 server 同节点），baseline `main 26f49e336a`（含 #58456）：output tok/s 984.7 → 995.2（+1.1%）、ITL avg/p50/p90 14.24/11.09/23.03 → 13.57/10.77/22.34（−4.7%/−2.9%/−3.0%）、TTFT p50 938 → 911（−2.8%）、server TPOT avg 13.57 → 12.93 ms（−4.8%），per-step verify time 47.1 → 45.7 ms，正文概述"TPOT is 4.8% lower"。

## 验证与回归风险

`pytest tests/kernels/quantization/test_rocm_mxfp8_linear.py -q`（MI355X）83 passed、`pytest tests/kernels/attention/test_rocm_triton_attn_dsv4.py -q`（MI355X）95 passed；新增测试覆盖 7 个 V4.1 形状（含 K=576）对 fp32 reference 的一致性、per-row scales 不压缩、in-launch split-K 的 bitwise repeatability、首次 split-K 在 capture 内回退、`wo_a` 两种 scale 布局；`pre-commit run`（含 mypy）clean。GSM8K（TP4、DSpark x5、5-shot、conc 128、max_tokens=12288、greedy）两 arm 完全一致：flexible-extract 0.9727 ± 0.0045、strict-match 0.9735 ± 0.0044。正文注明测量 commit `3b46e2d306` 在 in-launch split-K 与 retune 之前，数字偏保守。

## 关联

正文明确列出：#58456（grouped MXFP8 `wo_a`，baseline main 已含）、#57857（open，改同一 kernel、mask K tail 并加 `input_quant_key()`，互补且都动 `rocm_native.py`）、ROCm/aiter#5750（open，同思路的 AITER GEMM）、#48427 / #57760（re-quantize MXFP8 → FP8 PTPC，改变权重，不同取舍）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
