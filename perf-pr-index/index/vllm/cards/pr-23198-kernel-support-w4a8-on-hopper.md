# PR #23198: [kernel] Support W4A8 on Hopper

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @czhu-cohere |
| 合并 | 2025-08-24 |
| 分类 | Quantization \| tier: major |
| Label | performance, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注「4 file(s) dropped from diff for size (see PR for full diff)」 |
| 链接 | https://github.com/vllm-project/vllm/pull/23198 |

## 一句话结论

在 Hopper 上新增 CUTLASS W4A8 kernel，让 W4A8 接近 FP8 峰值吞吐并保持 W4A16 快速 decode。

## 动机

正文 Purpose：为 vLLM 增加基于 CUTLASS 的 W4A8 kernel（Hopper），参考 CUTLASS example 55，用 LUT trick 绕过 GEMM mainloop 里的 int4 → bf16 → fp8 转换。目的是提升 compute-bound 性能，让 W4A8 接近峰值 FP8 吞吐，同时保持 W4A16 的快速 decode 速度。

## 优化点

kernel 计算 `out = (w_q.to(scale_type) * w_s) @ a * s_a * s_c`：out 为 bf16，`w_q` packed int4，`scale_type`/`w_s`/`a` 均 fp8 e4m3（`w_s` group size 128、`a` 动态量化），`s_a` per-tok fp32、`s_c` per-channel fp32，per-tok/per-chan scaling 在 epilogue 完成。zp / activation reordering / 更小 group size 暂不支持；权重要求 signed int4（与常见 int4b8 不同，可无损转换），由 `cutlass_pack_scale_fp8`、`cutlass_encode_and_reorder_int4b` 处理布局。主文件 `w4a8_mm_entry.cu` 实现 prepack/weight shuffling、CUTLASS 模板与按 shape 的 dispatch，`mm_dispatch` 启发式来自 Llama 8/70/405B 的 tile/cluster sweep，aggregate 达各 shape 最佳配置的 ~1-2% 以内。新增 3 个 torch op 及 cudagraph fake 变体；前端 `_is_fp8_w4a8_sm90` 返回 `CompressedTensorsW4A8Fp8` scheme，`CutlassW4A8LinearKernel` 包装 per-tok 激活量化与 w4a8 op。正文还给出触发 W4A8 的 quantization config 示例：`compressed-tensors`、权重 pack-quantized（8 个 4-bit 值打包进 int32）、group size 128、激活 fp8 e4m3 动态 token 量化，权重与激活均对称，并 `ignore: ["lm_head"]`。

## 关键改动

- `csrc/quantization/cutlass_w4a8/w4a8_mm_entry.cu`：W4A8 主 kernel（prepack、CUTLASS 模板、mm_dispatch）。
- `csrc/torch_bindings.cpp`、`vllm/_custom_ops.py`：注册 3 个 torch op 及 fake 变体。
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`、`schemes/__init__.py`：`_is_fp8_w4a8_sm90` 与 `CompressedTensorsW4A8Fp8`。
- `vllm/model_executor/layers/quantization/kernels/mixed_precision/cutlass.py`（正文点名）：`CutlassW4A8LinearKernel`。
- 测试/基准：`test_cutlass_w4a8.py`、`benchmark_machete.py`、`weight_shapes.py`。

```text
 from .compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4
+from .compressed_tensors_w4a8_fp8 import CompressedTensorsW4A8Fp8
 from .compressed_tensors_w4a8_int import CompressedTensorsW4A8Int
```

## 性能声明

kernel perf benchmark（`benchmark_machete.py --act-type float8_e4m3fn ... model_bench`，Hopper；表格数值单位 PR 未注明）节选（shape | fp16 matmul | cutlass fp8 | machete w4a8 | cutlass w4a8）：(1x12288x14336) L=2 为 270.6 / 129 / 88.3 / 94.5；(32x12288x14336) L=2 为 240.4 / 141.7 / 97.3 / 90.5；(1024x12288x73728) L=1 为 2381.5 / 1526.2 / 1846.2 / 1354.2；(4096x12288x73728) L=1 为 9589.9 / 7706.7 / 7638.7 / 5093.2。serving benchmark（tp1/tp2、10k in/1k out、并发 1/4/8/16/32）结果仅截图，正文未给可读数字。E2E quality：gsm8k w4a16 0.8483699773 (1119/1319) → w4a8 0.8476118271 (1118/1319)，mmlu_pro 0.6907413564 → 0.6760305851。正文注明 "more detailed perf/quality evals pending"，regex 未检出定量加速比。

## 验证与回归风险

验证：`pytest tests/kernels/quantization/test_cutlass_w4a8.py` 通过（kernel 正确性 + cudagraph，参考用 fp8 fast accumulate）；kernel perf 与 serving benchmark、gsm8k / mmlu_pro 质量对照。可能受影响路径：Hopper SM90 的 compressed-tensors W4A8 config、cudagraph（fake op）、signed int4 权重编码；正文 TODO 列出更全面 quality evals、llm-compressor recipe、fp32 channel scales、fp16 output、更多 tile/cluster codegen、prepack 优化，并自述 zp/activation reordering/小 group size 尚不支持。

## 关联

PR 正文未给出同分类其它 vLLM PR 编号、前序或后续 PR（仅链接 CUTLASS example 55），无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
