# PR #728: Revert "fix: fp8 config"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#728](https://github.com/sgl-project/sglang/pull/728) |
| 作者 | Ying1123 |
| 合并日期 | 2024-07-25 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
回退 #723:删除针对 Llama3-405B fbgemm_fp8 的专用处理(强制 KV 头数=8 与权重加载时压缩 KV 头),恢复通用加载路径。

## 问题现象
正文仅一句 "Reverts sgl-project/sglang#723",未描述具体现象。被回退的原 PR 标题为 "fix: fp8 config",本 PR 将其撤销;具体症状与复现条件正文均未给出。

## 根因
正文未披露根因,仅说明是对 #723 的 revert。(由 diff 推断)机制:#723 曾在 `managers/controller/model_runner.py` 加入一组 Llama 3.1 405B FP8 专用逻辑——`is_llama3_405b_fp8()` 按 architectures/hidden_size=16384/intermediate_size=53248/126 层/`fbgemm_fp8` 识别该模型,命中时把 `num_key_value_heads` 强制改为 8;并 monkey patch `QKVParallelLinear.weight_loader`(`get_weight_loader_srt`),当 K/V shard 加载形状等于 `head_size * total_num_kv_heads * 2` 时用 `get_original_weight` 压缩掉重复的 KV 头。本次回退把这套特判整体删除,恢复通用加载路径。

## 修复方案
改动文件 `python/sglang/srt/managers/controller/model_runner.py`:
- 删除 `is_llama3_405b_fp8` 及 `load_model` 中对 `hf_config.num_key_value_heads = 8` 的强制覆盖;
- 删除 `get_original_weight`、`get_weight_loader_srt` 及 `QKVParallelLinear.weight_loader` 的 monkey patch(setattr 两行);
- 清理不再使用的导入(`Parameter`、`QKVParallelLinear`)。

关键 diff:
```diff
-def is_llama3_405b_fp8(model_config):
-    if (
-        model_config.hf_config.architectures[0] == "LlamaForCausalLM"
-        and model_config.hf_config.hidden_size == 16384
-        and model_config.hf_config.intermediate_size == 53248
-        and model_config.hf_config.num_hidden_layers == 126
-        and model_config.hf_config.quantization_config["quant_method"] == "fbgemm_fp8"
-    ):
-        return True
-    return False
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,即 #723 合入后至本回退落地之间。
- 回归面:只影响 `managers/controller/model_runner.py` 的模型加载路径,主要涉及 Llama 3.1 405B fbgemm_fp8 部署——该场景不再强制 num_key_value_heads=8、不再压缩 KV 权重,回到通用 fp8 加载行为;其他模型原本不命中特判,行为不变。K/V 权重形状异常的场景失去该 workaround 的兜底。

## 关联
- [sgl-project/sglang#723](https://github.com/sgl-project/sglang/pull/723)(正文注明的被 revert 对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
