# PR #6996: [Kernel] Tuned int8 Cutlass Kernels for SM75 (T4)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @varun-sundar-rabindranath |
| 合并 | 2024-07-31 |
| 分类 | Quantization \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/6996 |

## 一句话结论

为 SM75/T4 增加 4 组 tuned int8 GEMM 配置，小 M[1,64] 提速 20%。

## 动机

PR 未注明改前的具体瓶颈。正文仅说明 "Add tuned Int8 kernels for SM75 (T4)"，并指出新增 4 组 unique Gemm configs 以按 M 维度对 GEMM shapes 分层（stratify Gemm shapes along the M dimensions）。

## 优化点

在 SM75（T4）的 int8 CUTLASS scaled_mm 路径中新增 4 组 GEMM 配置，按 M 维度分区选用不同配置，取代原有的单一配置。正文给出的分区结论：M 在 [1, 64] 有 20% speedup；M 在 (64, 128] 与 main 相同；M 在 (128, 256] 部分 GEMM shape 最多有 3% 性能下降，但 PR 依据整体 perf runs 选择了更优的整体配置；M 在 (256, inf) 与 main 相同。

## 关键改动

```text
csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm75_dispatch.cuh: 按 M 维度分层的 SM75 int8 GEMM 配置分发
csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu: SM75 int8 scaled_mm 的 kernel 入口
benchmarks/cutlass_benchmarks/w8a8_benchmarks.py: 对应 int8 model_bench 基准
```

## 性能声明

正文 "Perf. Takeaways" 声明 M in [1, 64] - 20% speedup；M in (64, 128] - Same as main；M in (128, 256] - up to 3% perf. drop for some Gemm shapes；M in (256, inf) - Same as main。测试条件：Machine T4 x 1；命令 `python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype int8 model_bench --batch-sizes {1,16,32,64,128,256,512}`；表格覆盖 torch.int8 的 meta-llama/Llama-2-7b-hf-TP1、Llama-3-8b-TP1、Llama-2-13b-hf-TP1、Llama-2-70b-hf-TP1，列 cutlass main (us)、PR (us)、pytorch fp16 no-scales (us)、speedup vs cutlass main、speedup vs pytorch fp16。代表性数值：Llama-2-7b MKN=(1x4096x12288) 由 349.3 降到 276.7 us，speedup vs cutlass main 1.26、vs pytorch fp16 1.40；MKN=(128x4096x4096) 为 147.9 / 147.7 us、1.00。精度为 int8 W8A8，对比 pytorch fp16。

## 验证与回归风险

PR 未注明引入或依赖的单元测试，正文的 benchmark 命令与数据表即为验证。回归风险：正文明确 M 在 (128, 256] 的部分 GEMM shape 最多有 3% 性能下降，其余区间持平；未说明对其他硬件或精度的回归评估。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
