# PR #32414: [MoE Refactor] Oracle Select FP8+NVFP4 Kernels In Priority

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @robertgshaw2-redhat |
| 合并 | 2026-01-21 |
| 分类 | MoE \| tier: major |
| Label | documentation, performance, rocm, ready, ci/build, v1, llama, qwen, gpt-oss, nvidia, ready-run-all-tests |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 55 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/32414 |

## 一句话结论

重构 MoE kernel oracle，引入 kernel 优先级与标准构造接口，支持按硬件/模型自动选择 FP8+NVFP4 kernel。

## 动机

PR 正文 Purpose 指出，重构前 oracle 没有 kernel 优先级的概念，各 kernel 的构造接口不统一，也没有注册其所支持的特性。这带来三个问题：无法跨硬件与模型架构自动选择 kernel（因而无法 opt-in TRTLLM 一类 kernel）、只能在运行时而非初始化时才发现 kernel 是否支持当前部署（错误信息不清晰）、代码复用差。本 PR 即为解决这些问题而做的 MoE 框架重构。

## 优化点

正文列出三项机制改动：为 oracle 增加 kernel 优先级（priority）的概念；为 kernel 增加统一的标准构造接口（standard interface for construction）；为 kernel 增加所支持特性（supported features）的注册。由此实现跨硬件与模型架构的 kernel 自动选择、在 init 阶段校验 kernel 支持度以获得更清晰的错误、提升代码复用。本 PR 属于框架重构，PR 正文未说明具体算法或内核层面的性能改动，也未给出性能数字。

## 关键改动

pack 的 diff 因 55 file(s) dropped from diff 只保留了 benchmarks、tests 与 docs 类文件，核心源码文件未在素材中提供。可见的核心文件与作用：

```text
docs/design/moe_kernel_features.md: 新增 MoE kernel features 设计文档，配合 kernel 特性注册（diff 中新增）
tests/kernels/moe/modular_kernel_tools/common.py、mk_objects.py: 更新 modular kernel 测试工具以适配新的统一构造接口
benchmarks/kernels/benchmark_cutlass_moe_fp8.py、benchmark_cutlass_moe_nvfp4.py、benchmark_grouped_gemm_cutlass.py: benchmark 改用 make_dummy_moe_config 构造 FusedMoEExperts，替代散落的 out_dtype/e/n/k/device 参数
tests/evals/gsm8k/configs/moe-refactor/ 与 moe-refactor-dp-ep/ 下的多个 yaml: 新增/更新 Llama-4-Scout、Qwen3-30B-A3B FP8/NvFp4 等组合的 GSM8K eval 配置
```

```python
CutlassExpertsFp8(
    moe_config=make_dummy_moe_config(
        num_experts=num_experts,
        hidden_dim=k,
        intermediate_size_per_partition=n,
        in_dtype=a.dtype,
    ),
    quant_config=quant_config,
)
```

（摘自 `benchmarks/kernels/benchmark_cutlass_moe_fp8.py`：kernel 构造改用统一的 `moe_config`。）

## 性能声明

PR 正文未给出定量性能数据。pack 的 perf claims regex 未检测到数字；正文仅有 Purpose 与 Test Plan（"refactor tests in the CI/CD"），未附 benchmark 结果，PR 未注明测试条件。

## 验证与回归风险

PR 引入/依赖的测试：Test Plan 仅写明 "refactor tests in the CI/CD"；可见 diff 显示 `tests/kernels/moe/` 下多个单测与 `modular_kernel_tools`、`tests/evals/gsm8k` 配置随构造接口重构而更新，另有 `tests/compile/test_silu_mul_quant_fusion.py` 改动。PR 未注明回归风险，核心源码因 diff 被丢弃未包含在 pack，正文与可见 diff 未说明可能受影响的路径。

## 关联

无（PR 正文未给出其他 PR 编号线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
