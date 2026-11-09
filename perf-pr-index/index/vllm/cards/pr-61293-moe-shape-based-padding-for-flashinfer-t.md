# PR #61293: [MoE] Shape-based padding for FlashInfer TRT-LLM MXFP8 experts (gated and non-gated)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jhaotingc |
| 合并 | 2026-10-15 |
| 分类 | MoE \| tier: major |
| Label | ready, nvidia, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未见「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/61293 |

## 一句话结论

按 shape 给 MXFP8 experts 补零对齐，GB300 并发 64 上 W4A4+MXFP8-MoE 达 9640 tok/s/GPU（1.71× BF16）。

## 动机

FlashInfer TRT-LLM MXFP8 MoE kernel 要求 `intermediate % 128 == 0` 且 `hidden % 256 == 0`，否则报 `No valid config found`。本 PR 在该后端按 shape 对 MXFP8 experts 做 padding，gated 与 non-gated 都支持，使 expert 尺寸不对齐的 MXFP8-MoE checkpoint 能够运行；已对齐模型保持不变（no copy、no runtime padding）。

## 优化点

新增 `align_mxfp8_moe_weights_for_fi_trtllm`（紧邻 `align_moe_weights_for_fi`）：复用后者对 intermediate size 分别 pad gate 与 up 两半；hidden size 用零权重 padding，UE8M0 scales 同步用 127（即 scale 1.0）padding，使被补行列对结果无贡献，scales 可为 3D `[E, rows, cols]` 或 flattened，布局保持不变。`prepare_fp8_moe_layer_for_fi` 在 MXFP8 TRT-LLM 路径调用它并在调用点设置 `intermediate_size_per_partition` 与 `hidden_dim`；MoE runner 随后把 activations pad 到 padded hidden size 并 crop 输出。测试使用 `TrtLlmFp8ExpertsMonolithic`，而 `TrtLlmFp8ExpertsModular`（`trtllm_fp8_block_scale_routed_moe`）从不传 `activation_type`，因此恒为 gated。对齐量分别由 `round_up(n, 128)` 与 `round_up(k, 256)` 决定，新增测试据此断言 padded 后的 intermediate 与 hidden 尺寸，并覆盖 gated SiLU / non-gated ReLU² 及 1856×2688 等形状。

## 关键改动

- `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`：新增 `align_mxfp8_moe_weights_for_fi_trtllm` 及 `prepare_fp8_moe_layer_for_fi` 调用点，设置 padded 尺寸。
- `tests/kernels/moe/test_flashinfer.py`：新增 `test_trtllm_mxfp8_moe_pads_intermediate`，覆盖 gated SiLU 与 non-gated ReLU²、对齐与非对齐 shape（如 1856×2688），与 dequantized weights 上的 `fused_experts` 对比。

```text
+def align_mxfp8_moe_weights_for_fi_trtllm(
+    w13: torch.Tensor,
+    w2: torch.Tensor,
+    w13_scale: torch.Tensor,
+    w2_scale: torch.Tensor,
+    is_act_and_mul: bool,
+    min_alignment: int,
+    hidden_alignment: int,
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
+    """align_moe_weights_for_fi for MXFP8 weights and their UE8M0 block scales
+    for the FlashInfer TRT-LLM MoE kernel, which also pads the hidden size to a
+    multiple of hidden_alignment.
+
+    Padded weights are zero and padded scales are 127 (UE8M0 for 1.0), so the
+    padded rows and columns contribute nothing. Scales may be 3D
```

## 性能声明

条件：GB300、TP1、speed-bench 1k prompts / OSL 200、无 spec decode、并发 64（serve 标志见正文）。正文 Claims 与 Validation：一个 W4A4 + MXFP8-MoE checkpoint（本 PR 使能）达 **9,640 output tok/s/GPU**（189.6 tok/s/user），BF16 为 5,648（101.2），即 **1.71× throughput** 与 **1.87× per user**；正文亦称达到 BF16 的 1.71× output tok/s/GPU。Accuracy（GPQA-diamond，NeMo Gym，1x GB300、TP1、MTP=2、198 questions x 8 repeats、thinking on、temp 1.0、top_p 0.95）pass@1 为 76.45 ± 0.70%（无 prefix reuse）与 76.58 ± 0.45%（with prefix reuse）。

## 验证与回归风险

验证：`pytest tests/kernels/moe/test_flashinfer.py -k mxfp8` 在 B200 上 14 passed（新测试含 gated SiLU / non-gated ReLU²、对齐与非对齐 shape，及 4 个已有 MXFP8 TRT-LLM 测试）、GPQA 精度、上述 perf、`pre-commit run --files`（含 mypy）通过。可能受影响路径：FLASHINFER_TRTLLM MXFP8 MoE、gated/non-gated experts、Modular 路径（恒 gated）；已对齐模型正文称不变（无 copy、无 runtime padding），但 padding 引入权重与 scales 的加载期改写。

## 关联

PR 正文未给出同分类其它 PR 编号、前序或后续 PR，无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
