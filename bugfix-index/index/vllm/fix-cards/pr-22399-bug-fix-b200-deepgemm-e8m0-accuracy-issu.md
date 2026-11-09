# PR #22399: [Bug] Fix B200 DeepGEMM E8M0 Accuracy Issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22399](https://github.com/vllm-project/vllm/pull/22399) |
| 作者 | yewentao256 |
| 合并日期 | 2025-08-07 |
| labels | bug |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

B200 上 DeepGEMM E8M0 重量化未启用 UE8M0 scale 导致精度崩坏,补 `use_ue8m0=True` 修复。

## 问题现象

在 B200 上开启 `VLLM_USE_DEEP_GEMM=1` 运行 FP8 模型 `Qwen/Qwen3-30B-A3B-FP8`(`max_model_len=32768,enforce_eager=True`),用 lm_eval 测 gsm8k 5-shot:修复前 `flexible-extract` 与 `strict-match` 的 exact_match 均为 **0**(完全不可用);修复后分别为 0.8597 和 0.8939。属于典型的大规模精度/正确性问题。

## 根因

正文未披露具体机制,只说明是 B200 E8M0 精度问题。(由 diff 推断)`vllm/model_executor/layers/quantization/utils/fp8_utils.py` 的 `requant_weight_ue8m0_inplace` 在把权重按 2 的幂(UE8M0)scale 重新量化回 FP8 时,调用 `per_block_cast_to_fp8(w_dq, [block_m, block_k])` 未传 `use_ue8m0=True`,重量化产生的 scale 与 DeepGEMM 在 B200 上要求的 E8M0 格式不一致,导致权重数值偏离,推理精度崩坏。

## 修复方案

改 `vllm/model_executor/layers/quantization/utils/fp8_utils.py` 的 `requant_weight_ue8m0_inplace`:重量化调用时显式启用 UE8M0 模式,确保重写回的权重与 scale 符合 E8M0 约定。

```diff
         # Re-quantise using power-of-two scaling (UE8M0).
-        w_requant, s_requant = per_block_cast_to_fp8(w_dq, [block_m, block_k])
+        w_requant, s_requant = per_block_cast_to_fp8(w_dq, [block_m, block_k],
+                                                     use_ue8m0=True)
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 `requant_weight_ue8m0_inplace` 这条权重原地重量化路径(DeepGEMM/UE8M0 场景下的 FP8 权重写回)。改后重量化 scale 强制为 2 的幂,数值上是对该函数输出的修正;其他 cast 路径不受影响。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
