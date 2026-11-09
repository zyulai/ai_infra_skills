# PR #5183: [Kernel] Switch fp8 layers to use the CUTLASS kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tlrmchlsmth |
| 合并 | 2024-06-07 |
| 分类 | Quantization \| tier: major |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/5183 |

## 一句话结论

fp8 layers 从 torch._scaled_mm 切到 CUTLASS fp8 kernels，e2e 性能提升 5-15%。

## 动机

PR 未注明改前的具体瓶颈。正文只说明"在受支持时"把 fp8 layers 从 `torch._scaled_mm` 切换到 vLLM 自带的 cutlass fp8 kernels，并指出切换的原因是已在 neuralmagic/Meta-Llama-3-8B-Instruct-FP8 上观察到 e2e 性能提升。PR 正文极为简短，主体仅这一句切换说明与一句性能观察，外加两个外部链接；其余内容均为 vLLM 的 PR checklist 模板，没有独立的 Purpose、Test Plan 或 Test Result 段落，也未列出切换所依赖的 kernel 支持条件。

## 优化点

改动集中在 fp8 量化层的 GEMM 后端选择：在 kernel 支持的情况下，不再走 PyTorch 的 `torch._scaled_mm`，而改调 vLLM 的 cutlass fp8 kernels。正文未展开 CUTLASS kernel 的实现细节、支持的 shape 范围或 dispatch 条件，仅说明切换本身，并把更细的跨 GEMM size 对比指向另一份材料。

## 关键改动

```text
vllm/model_executor/layers/quantization/fp8.py: fp8 量化层，把 GEMM 后端从 torch._scaled_mm 切换到 CUTLASS fp8 kernels
vllm/_custom_ops.py: 本 PR 同时改动的 custom op 入口文件
```

## 性能声明

正文声明 "5-15% improvement in e2e performance on `neuralmagic/Meta-Llama-3-8B-Instruct-FP8`"。测试条件：模型 neuralmagic/Meta-Llama-3-8B-Instruct-FP8，指标为 e2e performance；PR 未注明硬件、精度、batch / workload 等测试条件，正文仅给出一份快速 e2e benchmark 表格链接，并指向 PR #5144 查看不同 GEMM size 的对比。

## 验证与回归风险

PR 未注明引入或依赖的测试，正文未包含任何测试命令、测试数值表格或精度对比，也未说明可能受影响的模型、硬件或精度路径；切换以 "when supported" 为前提，但正文没有列出 CUTLASS kernel 的具体支持条件与 shape 范围。

## 关联

正文引用 PR #5144 作为跨不同 GEMM size 的性能对比材料。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
