# PR #18778: [Perf] Tunings for SM100 FP8 CUTLASS kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-06-04 |
| 分类 | Quantization \| tier: major |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/18778 |

## 一句话结论

为 SM100 FP8 CUTLASS kernel 补 small M<128 配置，B200/Llama 8B GEMM 提速 1.7-2.5x。

## 动机

正文指出 Blackwell 的 FP8 CUTLASS kernel 原本只有一套 default configs，小 M 场景没有专门调优，因此本 PR 为 small M < 128 新增 configs。

## 优化点

在 SM100 FP8 CUTLASS kernel 的 dispatch 头文件里补齐小 M 区间的 kernel 配置，使 M < 128 的 GEMM 不再共用 default 配置。正文未展开具体 tile / warp 参数取值，只给出调优前后 `bench_fp8_gemm.py` 的 TFLOP/s 对照表；改动集中在单一 dispatch 文件，未涉及其他算子路径。对照表按 N=6144 K=4096、N=4096 K=4096、N=28672 K=4096、N=4096 K=14336 四组 GEMM shape 分别列出，batch_size 覆盖 1 到 1024，并区分 torch-bf16、fp8-tensor-w-tensor-a、fp8-channel-w-token-a 及其 noquant 变体的 TFLOP/s。

## 关键改动

```text
csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh: 为 SM100（Blackwell）FP8 scaled_mm 增加 small M < 128 的 kernel 配置分发
```

## 性能声明

正文声明 "For Llama 8B on B200, these tunings offer a GEMM improvement of: 1.7 to 2.5x speedup at M<64; 1.1 to 1.3x speedup at 64<=M<128"。Kernel benchmarks 使用 PR #17126 的 `python benchmarks/kernels/bench_fp8_gemm.py --model meta-llama/Llama-3.1-8B-Instruct --tp-sizes 1`，给出 B200 original tunings 与 B200 new tunings 的 BF16 vs FP8 GEMMs TFLOP/s 表（N/K 组合 6144×4096、4096×4096、28672×4096、4096×14336，batch_size 1~1024）。代表性数值：N=6144 K=4096 且 batch_size=1 时 fp8-channel-w-token-a 由 2.072256 提升到 5.222792 TFLOP/s，batch_size=16 时由 39.819737 提升到 83.468890 TFLOP/s。精度为 FP8（W8A8）；硬件 B200；模型 Llama 8B / meta-llama/Llama-3.1-8B-Instruct。

## 验证与回归风险

Accuracy eval：`lm_eval --model vllm --model_args pretrained=RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto`，结果 gsm8k flexible-extract 0.7657 ± 0.0117、strict-match 0.7422 ± 0.0120。Kernel benchmarks 依赖 PR #17126 的脚本。PR 未注明新增单元测试，也未说明对其他模型、硬件或精度的回归评估。

## 关联

正文注明 kernel benchmarks 使用 PR #17126 提供的工具。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
