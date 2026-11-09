# PR #6693: [BugFix] Fix RoPE error in Llama 3.1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6693](https://github.com/vllm-project/vllm/pull/6693) |
| 作者 | WoosukKwon |
| 合并日期 | 2024-07-23 |
| labels | ready |
| 热度 | 👍 9 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
正文为空;从 diff 看,修复 Llama 3.1 RoPE:删除 131072 特例注入、支持 rope_type 键并以 "llama3" 取代 "extended"。

## 问题现象
正文未描述具体现象(正文为空)。按标题,Llama 3.1 存在 RoPE 错误;正文未给出报错信息、复现步骤或影响面说明。

## 根因
正文未披露根因(正文为空)。(由 diff 推断)旧逻辑在 `ModelConfig.__init__` 中对 `max_position_embeddings == 131072` 且未提供 `rope_scaling` 的模型强制注入 `{"type": "extended"}` 特例,且各处只识别 `rope_scaling["type"]` 键与 `"extended"` 类型名;当 HF 侧配置改用 `rope_type` 键与 `"llama3"` 类型名后,vLLM 的解析与模型实际配置脱节,导致 Llama 3.1 的 RoPE 处理出错。本 PR 删除该特例、同时接受 `type`/`rope_type` 键、并把类型名从 `"extended"` 统一为 `"llama3"`,与此推断吻合。

## 修复方案
- `vllm/config.py` `ModelConfig.__init__`:删除"max_position_embeddings==131072 且无 rope_scaling 就注入 `{"type": "extended"}`"的特例块。
- `vllm/config.py` `_get_and_verify_max_len`:先从 `rope_scaling` 解析 `type` 或 `rope_type`(两者都缺则抛 `ValueError: rope_scaling must have a 'type' or 'rope_type' key.`);放行集合从 `{su, longrope, extended}` 改为 `("su", "longrope", "llama3")`,yarn 分支按解析出的 rope_type 判断。
- `vllm/model_executor/layers/rotary_embedding.py` `get_rope`:`scaling_type` 同样兼容 `type`/`rope_type` 两个键;`"extended"` 分支更名为 `"llama3"`,仍走 `ExtendedRotaryEmbedding`:

```diff
-        scaling_type = rope_scaling["type"]
+        scaling_type = rope_scaling[
+            "type"] if "type" in rope_scaling else rope_scaling["rope_type"]
-        if scaling_type not in {"su", "longrope", "extended"}:
+        if scaling_type not in {"su", "longrope", "llama3"}:
             scaling_factor = rope_scaling["factor"]
-        if scaling_type == "extended":
+        if scaling_type == "llama3":
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:Llama 3.1 类模型的 RoPE 配置解析路径。配置里仍写 `"extended"` 的旧写法将不再被识别(需改为 `"llama3"`);`rope_scaling` 缺少 `type`/`rope_type` 键时由静默处理变为抛 `ValueError`;131072 特例删除后,该类配置不再被改写注入 rope_scaling。yarn/su/longrope 分支逻辑保持等价,仅键名解析更宽容。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
