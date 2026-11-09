# PR #39510: [Kernel] Support TRTLLM GEN NVFP4 MoE for non-512-aligned hidden dims via weight padding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @danielafrimi |
| 合并 | 2026-04-14 |
| 分类 | MoE |
| Label | ready, nvidia |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 4 个源码文件 + 1 个新增单测 |
| 链接 | https://github.com/vllm-project/vllm/pull/39510 |

## 一句话结论

通过加载期零填充权重,让 TRTLLM GEN NVFP4 MoE 内核支持非 512 对齐 hidden_dim,在 Nemotron-3-Nano-30B-A3B-NVFP4 上较 Cutlass 吞吐提升约 22%。

## 动机

TRTLLM GEN NVFP4 MoE 内核原先要求 `hidden_dim % 512 == 0`,否则 `_supports_shape` 直接拒绝,hidden_dim 非 512 对齐的模型(如 NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4,hidden 2688)只能退回性能更低的 Cutlass 后端。本 PR 在加载期把 MoE 权重零填充到对齐,使这类模型也能用上更快的 TRTLLM 内核。

## 优化点

机制分三层:(1) `_supports_shape` 从 `hidden_dim % 512 == 0` 放宽为恒 True,前提是权重已在加载期对齐;(2) 新增 `align_trtllm_fp4_moe_hidden_dim_for_fi`:把 w13/w13_scale/w2/w2_scale 零填充到下一个 256 的倍数(PR 说明无需 512 对齐),同时记录 `hidden_dim_unpadded` 并更新 `moe_config.hidden_dim`;(3) 运行时激活侧 padding 与输出截断复用既有 MoE runner 基础设施(`_maybe_pad_hidden_states`),`apply()` 内不做自定义 padding。非 256 对齐的维度会打 warning 并可能因激活切片而性能退化。另修复一个 modular `apply()` autotuning bug:autotuning 时原来错误地 `return hidden_states`,改为 `return`。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/flashinfer_utils.py: 新增 hidden_dim 零填充对齐函数
vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py: TRTLLM 路径接入 padding 并改写 hidden_dim
vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py: 放宽 _supports_shape;修复 autotune 返回值 bug
tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py: 新增 padding 单元测试
```

```python
+def align_trtllm_fp4_moe_hidden_dim_for_fi(
+    w13, w13_scale, w2, w2_scale, min_alignment: int = 256,
+):
+    num_experts, gate_up_dim, packed_hidden_size = w13.shape
+    hidden_size = packed_hidden_size * 2
+    padded_hidden_size = round_up(hidden_size, min_alignment)
+    if padded_hidden_size == hidden_size:
+        return w13, w13_scale, w2, w2_scale, hidden_size
+    padded_w13 = w13.new_zeros((num_experts, gate_up_dim, padded_hidden_size // 2))
+    padded_w13[:, :, :packed_hidden_size] = w13
+    ...
```

## 性能声明

模型 nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4,random 数据集 input 1024 / output 8192、96 prompts、max-concurrency 32、request-rate inf、temperature 0,Cutlass(旧默认)vs TRTLLM(新默认):

- Request throughput:0.85 → 1.04 req/s(+22.35%)
- Output token throughput:6,981.92 → 8,515.42 tok/s(+21.96%)
- Mean TTFT:799.33 → 651.69 ms(-18.47%);Mean TPOT:4.49 → 3.68 ms(-18.04%)
- Median TTFT 反而 +20.16%(544.43 → 654.19 ms),PR 未解释

lm-eval(GSM8K,reasoning off):TRTLLM flexible 0.8597 / strict 0.8446;Cutlass flexible 0.8658 / strict 0.8491,精度略有差异。

## 验证与回归风险

新增单测覆盖"已对齐时 no-op(不拷贝)"与"2688 → 2816 零填充、非零区域数值不变"两个场景;在 Blackwell 上以 `--moe-backend flashinfer_trtllm` 验证了 serving,并验证 hidden_size=4096(无需 padding)的 NemotronH 120B 行为不变。风险面:激活 padding/输出截断走 runner 通用路径,非对齐维度必有运行时切片开销(有 warning);精度上 TRTLLM 与 Cutlass 存在小幅 lm-eval 差异;显存因权重 padding 略增(PR 未量化)。

## 关联

同 NVFP4 MoE 区域:#25609(FBGEMM NVFP4 支持扩展到 dense 模型)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
