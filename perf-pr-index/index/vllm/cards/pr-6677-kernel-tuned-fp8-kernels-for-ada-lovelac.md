# PR #6677: [Kernel] Tuned FP8 Kernels for Ada Lovelace

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @varun-sundar-rabindranath |
| 合并 | 2024-07-29 |
| 分类 | Quantization \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)，pack 末尾标注 4 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/6677 |

## 一句话结论

L40S 上调优 SM89 cutlass FP8 kernel，小 M GEMM 最高 5.13x（llama3-8b）。

## 动机

PR 正文列出三个目标：Tune SM89 cutlass FP8 kernels for performance、Refactor/Cleanup cutlass2x kernel code，以及 High-level performance takeaways（指出 M <= 16 的 GEMM 形状能获得较好的加速）。diff 显示原 `scaled_mm_c2x.cu` 单文件 470 行把 SM80/SM89 的 include、arch 门控包装、各 tile shape 的 `cutlass_2x_gemm` 特化与 epilogue dispatch 全部内联，本 PR 将其拆分重构。

## 优化点

调优 SM89（Ada Lovelace）FP8 内核，并把 `scaled_mm_c2x.cu` 由 470 行缩为 16 行的 thin wrapper：把 cute/cutlass include、`enable_sm75_to_sm80` / `enable_sm80_to_sm89` / `enable_sm89_to_sm90` 等 `__CUDA_ARCH__` 门控包装、以及 SM80/SM89 各 tile shape 的 `cutlass_2x_gemm` 特化抽到 `scaled_mm_c2x.cuh`、`scaled_mm_c2x_sm80_dispatch.cuh` 与新增的 `scaled_mm_c2x_sm89_dispatch.cuh`；sm89 epilogue 改走 `vllm::cutlass_gemm_sm89_dispatch<...>`。旧 SM80 分支可见 `GemmShape<128,128,64>`、`<64,128,128>`、`<32,64,128>`、`<16,64,128>` 等 tile shape。SM89 dispatch 头在 .cu 中被 include，但 pack 的 diff 未包含该文件（4 files dropped）。

## 关键改动

```text
csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu: 470→16 行，重构为 thin wrapper，引入 3 个 .cuh 头。
csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh: 抽出通用 cutlass_2x_gemm 模板与 enable_* arch 门控。
csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm80_dispatch.cuh: SM80 dispatch。
scaled_mm_c2x_sm89_dispatch.cuh: SM89 (Ada) 调优 dispatch（被 dropped，仅由 .cu include 引用）。

  -#include "cute/tensor.hpp"
  -#include "cutlass/gemm/device/gemm_universal_adapter.h"
  +#include "scaled_mm_c2x.cuh"
  +#include "scaled_mm_c2x_sm80_dispatch.cuh"
  +#include "scaled_mm_c2x_sm89_dispatch.cuh"
```

## 性能声明

测试条件：`Benchmark Serving` 机器 L40S x 1，模型 `neuralmagic/Meta-Llama-3-8B-Instruct-FP8`，`python benchmarks/benchmark_serving.py --backend openai --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1 --num-prompts 200 --port 8000`；`Gemm benchmarks` 同为 L40S x 1，`python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 model_bench --batch-sizes {1,16,32,64,128,256,512}`，dtype `torch.float8_e4m3fn`。Serving（Cutlass vs Pytorch kernels，200 prompts）：duration 201.75 vs 202.11 s，input tok/s 211.44 vs 211.07，output tok/s 200.13 vs 201.95；Mean TTFT 28.72 vs 30.49 ms、Median 28.00 vs 28.90、P99 58.57 vs 60.36；Mean TPOT 15.63 vs 16.68 ms、Median 15.57 vs 16.57、P99 18.71 vs 20.22；Mean ITL 15.57 vs 16.61 ms。High-level takeaways（min/max speedup；<1.0x 与 <0.9x 的 GEMM 形状数 /28）：llama2-7b 0.77x / 4.8x，11/28、2/28；llama3-8b 0.88x / 5.13x，8/28、1/28；llama2-13b 0.4x / 4.74x，9/28、3/28，M<=128 几乎总 >1.0x；llama2-70b 0.82x / 3.0x，7/28、2/28，M<=128 几乎总 >1.0x。GEMM 表（us，pytorch best / cutlass / speedup）示例：Llama-2-7b (1x11008x4096) 76.40/15.70/4.87、(32x4096x22016) 44.00/57.20/0.77；Llama-3-8b (1x14336x4096) 98.00/19.10/5.13、(256x4096x28672) 224.00/253.70/0.88；Llama-2-13b (1x13824x5120) 94.80/20.00/4.74、(512x5120x15360) 201.40/274.00/0.74；Llama-2-70b (1x8192x8192) 58.10/19.30/3.01。

## 验证与回归风险

Serving benchmark 以 Cutlass 与 Pytorch kernels 各跑 200 prompts 对比；GEMM benchmark 覆盖 4 个模型各 28 个形状。回归风险：多个 GEMM 形状性能低于 1.0x（llama2-7b 达 11/28、llama2-13b 最低 0.4x），PR 正文明确指出加速集中在 M <= 16 的小 M 场景，大 M 常回退；llama2-13b/70b 在 M<=128 几乎都 >1.0x。重构涉及 SM80/SM89 arch 门控与 epilogue dispatch，可能影响非 Ada (SM80) 路径及 int8/bf16/half 等类型。PR 正文 checklist 要求通过 linter 与测试，但未给出具体单测结果，也未注明 CI 回归套件。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
