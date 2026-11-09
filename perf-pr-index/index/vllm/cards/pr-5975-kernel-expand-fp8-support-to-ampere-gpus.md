# PR #5975: [Kernel] Expand FP8 support to Ampere GPUs using FP8 Marlin

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2024-07-03 |
| 分类 | Quantization（tier: major） |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「10 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/5975 |

## 一句话结论

FP8 Marlin 把 FP8 量化扩展到 Ampere（SM 8.0/8.7），memory-bound 场景最高提速 2x、权重显存省 2x。

## 动机

PR 正文指出，vLLM 原有的 FP8 支持只覆盖具备硬件 FP8 能力的 GPU（Hopper 与 Ada Lovelace），无原生 FP8 支持的 Ampere 卡无法使用。该工作引入 FP8 Marlin——一个用于 FP8→BF16/FP16 转换的快速融合反量化内核，把 FP8 量化扩展到更广泛的 GPU，目标是让没有硬件 FP8 的用户也能获得 FP8 带来的显存与性能收益。正文强调这是 weight-only 方案，与既有 W8A8 FP8 不同，因激活无需量化而精度更高。

## 优化点

关键点是复用 GPTQ Marlin 内核中已有的 8-bit 整数支持：把 FP8 权重打包进 int32 doublewords（GPTQ 格式），再排列成 Marlin 布局，并用位运算与 SIMT 操作实现高效的 4xFP8 → 4xFP16/BF16 反量化。这样在 Ampere（SM 8.0 与 8.7）上也能跑 FP8 权重，权重显存减半从而支持更大 batch，且精度与 FP16 baseline 相当。使用方式很简单：运行时指定 `quantization="fp8"`，或加载预量化的 FP8 checkpoint。正文还说明当前正把 scale 扩展为 channelwise，未来工作会回退到 per-tensor scale，且不含 MoE 模型支持。

## 关键改动

```text
csrc/quantization/fp8/fp8_marlin.cu: 新增 FP8 Marlin GEMM CUDA 内核，基于 GPTQ Marlin 实现 FP8→FP16/BF16 融合反量化
csrc/ops.h: 新增 fp8_marlin_gemm 算子声明
```

```cpp
// csrc/ops.h
torch::Tensor fp8_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                              torch::Tensor& b_scales, torch::Tensor& workspace,
                              int64_t num_bits, int64_t size_m, int64_t size_n,
                              int64_t size_k);
// csrc/quantization/fp8/fp8_marlin.cu
#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");
```

## 性能声明

PR 正文的 Key features 与结论段给出以下声明：在 memory-bound 场景性能提升最高 2x（"Improves performance up to 2x in memory-bound scenarios"）；权重显存占用减少 2x，从而允许更大 batch；精度与 FP16 baselines 相当。测试条件：正文注明 "Tested on H100, A100, and A10 GPUs"，并附 A10 / A100 的 E2E Latency、GSM8k lm-eval、以及 A10 / A100 的 layer-wise sweep（PyTorch FP16 vs FP8 Marlin MatMul）截图；正文指出在显存带宽更低的 GPU（A10、RTX 3090）与更大模型上收益更高。具体模型名称、精度配置与 workload 参数 PR 未注明测试条件，且正文未给出可读的具体数值（benchmark 结果以图片形式给出）。

## 验证与回归风险

验证方面，正文仅给出 "Tested on H100, A100, and A10 GPUs" 及上述 E2E / GSM8k / layer-wise 截图链接，未列出具体测试命令或数值。可能受影响的路径：diff 新增 `fp8_marlin_gemm` 算子与 `fp8_marlin.cu` 内核，作用于 `quantization="fp8"` 与预量化 FP8 checkpoint 的加载与 GEMM 路径；CUDA 源码中带 `__CUDA_ARCH__ < 800` 的分支直接报 "marlin_gemm(..) requires CUDA_ARCH >= 8.0"，故 SM 8.0 以下设备不可用；`static_assert` 限定只支持 float16 / bfloat16。正文注明不含 MoE 模型支持、scale 形态仍在调整，属功能与回归边界。其余回归信息 PR 未注明。

## 关联

无（正文引用 HuggingFace assisted-generation 博客与 transformers 实现作为算法来源，未指向 vLLM 内的前序/后续 PR）

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
