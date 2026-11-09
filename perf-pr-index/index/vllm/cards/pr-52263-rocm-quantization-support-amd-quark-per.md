# PR #52263: [ROCm][Quantization] Support AMD Quark per-block FP8 for fused MoE layers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jimmy-adams |
| 合并 | 2026-09-08 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, ready, needs-rebase, quantization, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 "file(s) dropped from diff" 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/52263 |

## 一句话结论

让 Quark per-block FP8 用于 fused MoE，MI355X 上 drafter 加载成功、acceptance 94.2%。

## 动机

#47972 为 Quark linear 层加入 per-block(128x128) FP8，但 `QuarkW8A8Fp8MoEMethod` 仍拒绝该 qscheme：routed experts 按 per-block 量化的 Quark checkpoint 加载完所有 linear 层后失败，抛 `ValueError: For FP8 Fused MoE layers, only per-tensor and per-channel scales... Found per_block, per_group`。本 PR 补齐 fused MoE 侧缺口。

## 优化点

`QuarkW8A8Fp8MoEMethod` 现接受 `per_block`，遵循 `CompressedTensorsW8A8Fp8MoEMethod` 的 `BLOCK` 策略；per-tensor/per-channel 路径不动，仅 scale 布局不同。`create_weights` 在 128x128 block grid 上分配 weight scales 并打 `FusedMoeWeightScaleSupported.BLOCK` 标记，使 `RoutedExperts.weight_loader` 按 block 而非按 row 分片；scales 注册为 Quark 导出的 `w13_weight_scale`/`w2_weight_scale`，加载无需模型特定 checkpoint 映射。`__init__` 从 weight config 读 `block_size` 并要求动态 activation 量化，拒绝静态 input scale；`block_shape` 经 `_setup_kernel` 与 `get_fused_moe_quant_config` 传入 `make_fp8_moe_quant_config` 以选 block-scaled backend。三处 byte-identical 的 fused-MoE 对齐检查统一移到 `fp8_utils` 的 `validate_fp8_block_shape_moe`。

## 关键改动

```text
vllm/model_executor/layers/quantization/quark/quark_moe.py: QuarkW8A8Fp8MoEMethod 接受 per_block、按 block grid 分配/分片 scale、要求动态激活量化
vllm/model_executor/layers/quantization/utils/fp8_utils.py: 新增 validate_fp8_block_shape_moe，统一三处 byte-identical 的对齐检查
vllm/model_executor/layers/quantization/fp8.py: block_shape 穿到 make_fp8_moe_quant_config 以选 block-scaled backend
vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py: BLOCK 策略对齐检查复用来源
tests/quantization/test_quark.py: 新增 per_block MoE 单测（动态输入校验、权重形状、block 网格分片、错位分区拒绝）
```

关键代码（来自 `tests/quantization/test_quark.py` diff）：

```python
assert layer.w13_weight_scale.shape == (
    4,
    w13_num_shards * (256 // 128),
    512 // 128,
)
assert layer.w2_weight_scale.shape == (4, 512 // 128, 256 // 128)
# The loader shards block scales on the block grid, not per row.
for scale in (layer.w13_weight_scale, layer.w2_weight_scale):
    assert scale.quant_method == FusedMoeWeightScaleSupported.BLOCK.value
```

## 性能声明

PR 正文未给出吞吐/延迟类加速数字，Test Result 是端到端可用性与精度验证。单元 `pytest tests/quantization/test_quark.py -k per_block` 报告 `8 passed, 35 deselected`（覆盖四个新 MoE 测试与 #47972 的 per-block linear 测试）。End to end 条件：1x MI355X (gfx950)、`Qwen3.5-35B-A3B` 经 AMD Quark 重量化为 NVFP4 backbone 加 per-block FP8 MTP layer、TP=1、`num_speculative_tokens=1`；drafter 的 `mlp.experts.routed_experts` 绑定 `QuarkW8A8Fp8MoEMethod`，在 TRITON FP8 MoE backend 上无需模型特定 weight mapping 即加载，scales 按 block grid 分片为 `w13_weight_scale (256, 8, 16)` 与 `w2_weight_scale (256, 16, 4)`（weight_block_size=[128, 128]）；`Avg Draft acceptance rate` 平均 94.2%。同一 run 的 `lm_eval` GSM8K 5-shot、全 1319 题、启用 drafter、chat template 关闭 thinking：flexible-extract 与 strict-match 的 `exact_match` 均为 `0.9613 ± 0.0053`。

## 验证与回归风险

验证为上述单测与 1x MI355X 端到端加载/GSM8K 精度；正文未列 GPU CI 全量套件或新增回归路径说明。可能受影响路径：Quark FP8 fused MoE 的 scale 加载与分片、TRITON FP8 MoE backend、per-block 量化 MTP drafter 的 acceptance；正文声明 per-tensor/per-channel 路径未改动。

## 关联

#47972（Quark linear 层 per-block 128x128 FP8 前序实现，正文明确引用）。无其他明确 PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
