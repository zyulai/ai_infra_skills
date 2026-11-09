# PR #32614: fix: Add glm4_moe_lite to MLA detection

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32614](https://github.com/vllm-project/vllm/pull/32614) |
| 作者 | marksverdhei |
| 合并日期 | 2026-01-23 |
| labels | ready, v1, nvidia |
| 热度 | 👍 12 · 💬 13 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论

GLM-4.7-Flash(`glm4_moe_lite`)实际用 MLA 却不在 `is_deepseek_mla()` 检测名单里,回退普通 KV 缓存导致约 4 倍 KV 显存;补入检测并按 MLA 维度约束 SM100 后端选择。

## 问题现象

GLM-4.7-Flash(`glm4_moe_lite`)在 vLLM 中不会被识别为 MLA 模型:它通过 `Glm4MoeLiteMLAAttention`(继承自 `DeepseekV2MLAAttention`)使用 Multi-head Latent Attention,但模型架构不在 MLA 检测中。正文披露后果是回退到标准 KV caching 而非高效 MLA caching,KV cache 显存占用约高 4 倍。

## 根因

正文披露:`vllm/transformers_utils/model_arch_config_convertor.py` 的 `is_deepseek_mla()` 架构清单缺少 `glm4_moe_lite`(及其 MTP 变体 `glm4_moe_lite_mtp`),导致该 MLA 模型未被检测出来。另注:`glm4_moe` 被有意排除,因为它用的是标准注意力(`Glm4MoeAttention` + `vllm.attention.layer.Attention`)。

## 修复方案

1. `model_arch_config_convertor.py`:`is_deepseek_mla()` 列表中加入 `glm4_moe_lite`、`glm4_moe_lite_mtp`。
2. SM100 兼容性(正文:SM100 上该模型的多种 MLA decode/prefill kernel 有问题):新增 `is_deepseek_r1_mla_compatible()` 检查 DeepSeek R1 兼容 MLA 维度(`qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128`);`use_flashinfer_prefill()` 与 `use_trtllm_ragged_deepseek_prefill()` 在维度不兼容时返回 False;`flashinfer_mla.py` 的 `supports_combination` 在 `qk_nope_head_dim != 128` 时返回不兼容说明。
3. `platforms/cuda.py`:Blackwell 默认后端选择增加 `qk_nope_head_dim == 128` kernel 约束,不满足时按优先级回退 CUTLASS_MLA(强制 block_size=128)→ FlashMLA → Triton MLA。关键 diff:

```diff
+    "glm4_moe_lite",
+    "glm4_moe_lite_mtp",
     "kimi_k2",
```

作者用 `marksverdhei/GLM-4.7-Flash-fp8` 在 2x RTX 3090 验证:MLA 被正确检测、启用高效 KV caching,每卡显存 14.7 GB;SM100 上默认 CUTLASS_MLA decode + FA2 prefill,TRITON_MLA 亦测试可用。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:`is_deepseek_mla()` 新增两个架构名只影响 GLM-4.7-Flash 的检测;但 SM100 MLA 后端默认选择逻辑与 `supports_combination` 校验对所有 MLA 模型生效——非 R1 维度(qk_nope_head_dim != 128)的 MLA 模型在 Blackwell 上不再被强推 FlashInfer MLA,而改走 CUTLASS_MLA/FlashMLA/Triton 回退链,可能改变这些模型的默认后端与性能特征。DeepSeek R1 维度模型行为不变。

## 关联

正文未列出(正文提及合作者 @mgoin,无 issue/PR 链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
