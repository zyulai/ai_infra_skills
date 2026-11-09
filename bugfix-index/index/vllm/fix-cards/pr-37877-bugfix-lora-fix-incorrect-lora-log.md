# PR #37877: [Bugfix][LoRA] Fix incorrect LoRA Log

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37877](https://github.com/vllm-project/vllm/pull/37877) |
| 作者 | jeejeelee |
| 合并日期 | 2026-03-23 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
LoRA 加载警告误报可支持模块将被忽略;检查与提示的基准从 supported_lora_modules 改为 expected_lora_modules。

## 问题现象
加载 LoRA adapter(正文示例为 Qwen3-0.6B 的 `/mnt/data2/Models/Qwen3-0.6B-LoRA/self_cognition_Alice`)时,`worker_manager.py` 针对每一层打出大量 WARNING:`LoRA module 'model.layers.7.mlp.gate_proj' ... is not in the model's supported LoRA target modules [down_proj, embed_tokens, gate_up_proj, lm_head, o_proj, qkv_proj]. These parameters will be ignored, which may cause abnormal model behavior.`。被点名的 `gate_proj`/`up_proj`/`k_proj`/`q_proj`/`v_proj` 等实际属于(映射后的)受支持模块,警告为误报,还错误声称这些参数会被忽略、可能导致模型行为异常。正文只贴日志,未评估功能性影响。

## 根因
正文仅说明目的为修复上述日志输出,未展开根因。由 diff 推断:忽略警告的判断 `is_supported_lora_module(module_name, supported_lora_modules)` 以及警告文案里打印的模块列表,都用的是 `supported_lora_modules`(模型实际注册的合并形式模块名,如 `gate_up_proj`),而 adapter 里存的是拆分前的原始模块名(如 `gate_proj`),两者口径不一致,导致按原始名检查时被误判为「不支持」。

## 修复方案
改动 `vllm/lora/worker_manager.py` 的 `_load_adapter`:先取 `expected_lora_modules_lst = list(expected_lora_modules)`,警告判断与文案中的模块列表全部改用它。关键 diff 摘录:

```python
expected_lora_modules_lst = list(expected_lora_modules)
for module_name in lora.loras:
    if not is_supported_lora_module(module_name, expected_lora_modules_lst):
        logger.warning_once(
            "LoRA module '%s' in adapter '%s' is not in the "
            "model's supported LoRA target modules [%s]. "
            ...
            ", ".join(sorted(expected_lora_modules_lst)),
        )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅改变该条警告日志的触发条件与打印内容,LoRA 权重加载与推理逻辑不变;真正不被支持的模块仍会按新基准警告,提示文案更贴合用户 adapter 里的模块命名。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
