# PR #32688: [Quantization][Deprecation] Remove Marlin 24

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @robertgshaw2-redhat |
| 合并 | 2026-01-28 |
| 分类 | Quantization(major) |
| Label | performance, ready, ci/build, nvidia |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 6 个文件,pack 注明另有 15 个文件因长度省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/32688 |

## 一句话结论

按 0.14 弃用公告正式删除 2:4 structured sparsity 的 Marlin 24 量化内核与全链路代码。

## 动机

Marlin 24 已随 0.14 版本发布弃用公告,本 PR 在 0.15 落地实际移除(正文原话:"now that we have released 0.14 with the deprecation notice, actually deprecate in 0.15")。正文仅此一句动机,未再展开。

## 优化点

纯删除型 PR,无新逻辑。可见 diff 删除 `csrc/quantization/marlin/sparse/` 整目录(`marlin_24_cuda_kernel.cu`、`common/mma.h`、`common/mem.h`、`common/base.h` 与 Sparse-Marlin LICENSE),并从 `benchmarks/kernels/benchmark_marlin.py` 移除 `gptq_marlin_24_gemm` 基准分支、`GPTQ_MARLIN_24_*` 常量与 `marlin_24_quantize` 导入;pack 注明另有 15 个文件因 diff 过长省略(含 Python 侧量化层与测试,可见 import 显示 `vllm/model_executor/layers/quantization/gptq_marlin_24.py`、`marlin_utils_test_24` 均在被移除引用之列)。删除后 2:4 稀疏 Marlin kernel 树不再参与编译与维护。

## 关键改动

```text
csrc/quantization/marlin/sparse/(整目录删除): Marlin 24 CUDA kernel、tensor-core mma/mem/base 封装与 Sparse-Marlin LICENSE
benchmarks/kernels/benchmark_marlin.py: 移除 gptq_marlin_24_gemm 基准与 marlin_24 参数生成
其余 15 个文件(pack 省略): Python 量化层与测试,详见 PR 页
```

## 性能声明

无性能数字。本 PR 为弃用清理,正文与可见 diff 均未附 benchmark。

## 验证与回归风险

`benchmark_marlin.py` 中常规 marlin 与 gptq→marlin repack 基准分支保留,说明稠密 Marlin 路径不受影响。回归面:使用 2:4 structured sparsity 权重的 GPTQ checkpoint 自 0.15 起无法再用 Marlin 24 kernel,需改用 repack 到稠密 Marlin 或其他量化方案;相关测试与 CI job 随代码移除(label 含 ci/build)。

## 关联

素材未列出具体前序/后续 PR;可对照同批移除型 PR #65230(移除 QuTLASS 与 fp_quant)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
