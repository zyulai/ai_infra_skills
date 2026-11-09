# PR #8987: Fix incorrect default get_hidden_dim logic

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8987](https://github.com/sgl-project/sglang/pull/8987) |
| 作者 | lifuhuang |
| 合并日期 | 2025-08-09 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
LoRA 的默认 `get_hidden_dim` 对 q/kv/o_proj 维度计算错误;按 `head_dim` 重写推导逻辑,并清理 llama、gemma2 等模型里不再需要的模型级覆写。

## 问题现象
正文未描述具体现象,Motivation 仅写「See #8939」;Modifications 概括为两点:修正此前错误的计算逻辑、清理不必要的模型级覆写。影响面为 LoRA 加载时各目标模块的输入/输出维度计算。

## 根因
正文未展开根因细节,只指向 #8939 并称「Correct previously wrong calculation logic」。(由 diff 推断)旧逻辑存在两处问题:一是把 `q_proj`/`o_proj`/`qkv_proj` 的输入输出维度一律当作 `hidden_size`,把 `kv_proj` 输出维度算成 `hidden_size // (num_attention_heads // num_key_value_heads)`,在 `head_dim != hidden_size / num_attention_heads` 的模型(如 gemma2 的 `head_dim * num_attention_heads != hidden_size`)上不成立;二是这些模型各自维护了一份覆写实现,重复且不一致。新逻辑统一以 `head_dim`(缺省回落到 `hidden_size // num_attention_heads`)为基准推导各模块维度。

## 修复方案
1. 重写 `python/sglang/srt/lora/utils.py` 的 `get_hidden_dim`:
```diff
-        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
-            return config.hidden_size, config.hidden_size
-        elif module_name in ["kv_proj"]:
-            return config.hidden_size, config.hidden_size // (
-                config.num_attention_heads // config.num_key_value_heads
+        head_dim = getattr(
+            config, "head_dim", config.hidden_size // config.num_attention_heads
+        )
+        if module_name == "qkv_proj":
+            return (config.hidden_size, None)  # qkv_proj is only used in LoRA A
+        elif module_name == "kv_proj":
+            return (None, head_dim * config.num_key_value_heads)
+        elif module_name == "q_proj":
+            return (None, head_dim * config.num_attention_heads)
+        elif module_name == "o_proj":
+            return (head_dim * config.num_attention_heads, config.hidden_size)
```
`None` 标注该侧维度不适用(qkv_proj/q_proj 仅用于 LoRA A,kv_proj 仅用于 LoRA B)。
2. 删除 `llama.py`、`gemma2.py`、`granite.py`、`qwen3.py`、`torch_native_llama.py` 中各自的模型级 `get_hidden_dim`(及 llama/granite/gemma2/torch_native_llama 的 `get_module_name`)覆写,统一走共享默认逻辑;`gemma3n_mm.py` 保留覆写但同步改为新的 None 语义分支。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:所有走默认 `get_hidden_dim` 的 LoRA 目标模块维度计算,以及原先带自定义覆写的 5 个模型——它们此后行为完全由共享逻辑决定;head_dim 与 hidden_size/num_attention_heads 不一致的模型维度结果会与修复前不同。diff 注释注明 qkv 的特殊处理将在 #8940 中继续解决。

## 关联
- [#8939](https://github.com/sgl-project/sglang/issues/8939)(Motivation 指向该条)
- #8940(见 diff 内 TODO 注释,非正文链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
