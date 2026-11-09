# PR #24818: Fix: Correct FusedMoE layer reference in auto_round quantization

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24818](https://github.com/vllm-project/vllm/pull/24818) |
| 作者 | David-Wen2025 |
| 合并日期 | 2025-09-19 |
| labels | ready |
| 热度 | 👍 5 · 💬 9 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
auto_round 量化的 AWQ/GPTQ MoE marlin 分支引用不存在的 `layer.moe` 触发 AttributeError;改为引用 `layer.moe_config` 修复。

## 问题现象
使用 `auto-round` 量化加载带 FusedMoE 层的模型时启动失败,报:

```text
AttributeError: 'FusedMoE' object has no attribute 'moe'
```

报错位置:`vllm/model_executor/layers/quantization/auto_round.py` 第 330 行附近。复现环境:NVIDIA RTX PRO 6000 WorkStation 96G,vLLM v0.10.2(torch 2.8+cu129),`vllm serve ... --quantization "auto-round" --dtype "bfloat16" --max-model-len 131072`,模型为 Qwen3-Next-80B-A3B-Instruct-AutoRound。修复后 `Application startup complete`。

## 根因
PR 正文披露:FusedMoE 对象没有 `.moe` 属性,AWQ 与 GPTQ 的 MoE 方法在 use_marlin 分支里却引用了 `layer.moe`,应直接使用 layer;由此在使用 auto_round 量化 + FusedMoE 层时抛出 AttributeError(对应 issue #24803)。

## 修复方案
改动集中在 `vllm/model_executor/layers/quantization/auto_round.py`,共两处:`apply_awq_quant_layer` 与 `apply_gptq_quant_layer` 的 `isinstance(layer, FusedMoE)` 且 `use_marlin` 分支。正文表述为「把 `layer.moe` 改为 `layer`」,diff 实际把构造 MoE method 的第二参数从 `layer.moe` 改为 `layer.moe_config`:

```diff
             if use_marlin:
-                return AWQMoEMethod(quant_args_marlin, layer.moe)
+                return AWQMoEMethod(quant_args_marlin, layer.moe_config)
```

```diff
             if use_marlin:
-                return GPTQMarlinMoEMethod(quant_args_marlin, layer.moe)
+                return GPTQMarlinMoEMethod(quant_args_marlin, layer.moe_config)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(复现环境为 vLLM v0.10.2)。
- 回归面:仅影响 auto_round 量化下 FusedMoE + marlin 路径的 method 构造;其他量化方法与非 marlin 分支不变。作者验证:Qwen3-Next-80B-A3B-Instruct-AutoRound 启动成功,GSM8K(limit 250)flexible-extract 0.884 / strict-match 0.844,MMLU acc 0.8687;另测 Qwen3-Coder-30B-A3B-Instruct-int4-AutoRound、GLM-4.5-Air-AWQ-FP16Mix、Qwen3-32B-AWQ 均正常。

## 关联
- issue #24803(正文 Purpose 提到 "To fix the issue #24803",未给出链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
