# PR #21476: [Perf] Cuda Kernel for Int8 Per Token Group Quant, 15x faster compared with triton

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-07-26 |
| 分类 | Quantization \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/21476 |

## 一句话结论

新增 int8 per-token group quant CUDA kernel，H100 上比 Triton 快约 15x。

## 动机

PR 未注明改前的具体瓶颈。正文仅说明 "Add support for int8 per-token group quant CUDA kernel"，并标注它是 #21083 的后续 PR（Follow up pr）。

## 优化点

为 int8 per-token group quant 增加 CUDA kernel 实现，与 Triton 实现形成对照升级。正文未提供 kernel 的具体实现细节、分组策略或 dispatch 条件，只给出 `benchmark_per_token_group_quant.py` 的逐配置耗时对比；改动涉及 CUDA 源文件、头文件与 torch binding，以及 Python 侧 int8 工具。benchmark 分别给出 fp8 与 int8 两张表：fp8 表按 shape、group size、col_major、ue8m0、dtype 五个维度组合共 24 个配置，int8 表为 6 个配置，两者都以 CUDA 实现耗时对 Triton 实现耗时计算 speed-up。

## 关键改动

```text
csrc/quantization/compressed_tensors/int8_quant_kernels.cu: int8 per-token group quant 的 CUDA kernel
csrc/quantization/fp8/per_token_group_quant.cu: per-token group quant 相关 CUDA 实现
csrc/quantization/per_token_group_quant_8bit.h: 8bit 量化的公共声明
csrc/torch_bindings.cpp 与 csrc/ops.h: 注册并暴露新算子
vllm/model_executor/layers/quantization/utils/int8_utils.py: Python 侧 int8 量化工具
```

## 性能声明

正文声明 "Benchmark: **15x faster compared with triton**"，命令 `python benchmark_per_token_group_quant.py`（H100-GPU17）。fp8 表覆盖 shape (32,128)/(64,256)/(16,512)、gs=64/128、col_major=0/1、ue8m0=0/1、dtype=fp8，CUDA 0.010-0.012 ms 对 Triton 0.155-0.169 ms，speed-up ×13.59-×15.73；int8 表 6 个配置 CUDA 均为 0.010 ms 对 Triton 0.155-0.158 ms，speed-up ×15.33-×15.87。硬件 H100；精度为 int8 / fp8 per-token group quant。关于端到端，正文写道 "I don't know which model to test on, please let me know. But basically I think it would give similar e2e performance like fp8 (6% improvement for throughput)"——其中 6% 为作者推测、PR 未给出模型级实测数据。

## 验证与回归风险

Unit test：`pytest test_per_token_group_quant.py` 收集 30 items，30 passed in 6.38s（Python 3.12.11、pytest-8.4.1、pluggy-1.6.0）。benchmark 逐配置表为性能验证。PR 未注明可能受影响的模型、硬件或精度路径的回归评估。

## 关联

正文标注为 PR #21083 的后续（Follow up pr）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
