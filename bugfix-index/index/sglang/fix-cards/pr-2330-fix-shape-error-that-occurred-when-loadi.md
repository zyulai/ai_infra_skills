# PR #2330: Fix shape error that occurred when loading lora weight of gemma2 model.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2330](https://github.com/sgl-project/sglang/pull/2330) |
| 作者 | upskyy |
| 合并日期 | 2024-12-08 |
| labels | (none) |
| 热度 | 👍 6 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
gemma2 的 hidden_size 与 head_dim×heads 不等,LoRA 加载沿用 llama 维度假设导致 shape error,补 get_hidden_dim。

## 问题现象
加载 gemma2 模型的 LoRA 权重时出现 shape error(标题与正文),影响 gemma2 的 multi-LoRA 推理;作者验证修复后 gemma2 多 LoRA 推理工作正常。

## 根因
正文披露:llama 与 gemma2 的 `get_hidden_dim` 语义不同。llama 中 `head_dim * num_attention_heads` 等于 `hidden_size`(3072 = 128 * 24),用 `self.config.hidden_size, self.config.hidden_size` 没问题;而 gemma2 中两者无关,`hidden_size` 为 2304、`head_dim * num_attention_heads` 为 2048(2304 != 2048)。且 gemma2 模型当时未实现 `get_hidden_dim`,从而绕过了 lora_manager.py(L34-46)的相应代码,导致按错误维度加载 LoRA 权重。

## 修复方案
`python/sglang/srt/models/gemma2.py` 新增 `get_hidden_dim(module_name)`(返回 input_dim, output_dim):q_proj/qkv_proj 为 `(hidden_size, head_dim * num_attention_heads)`;o_proj 为 `(head_dim * num_attention_heads, hidden_size)`;kv_proj 为 `(hidden_size, head_dim * num_key_value_heads)`;gate_up_proj 为 `(hidden_size, intermediate_size)`;down_proj 为 `(intermediate_size, hidden_size)`;其余 `raise NotImplementedError()`。另新增 `get_module_name`(q/k/v_proj→qkv_proj,gate/up_proj→gate_up_proj)。关键摘录:

```diff
+    def get_hidden_dim(self, module_name):
+        # return input_dim, output_dim
+        if module_name in ["q_proj", "qkv_proj"]:
+            return (
+                self.config.hidden_size,
+                self.config.head_dim * self.config.num_attention_heads,
+            )
+        elif module_name in ["o_proj"]:
+            return (
+                self.config.head_dim * self.config.num_attention_heads,
+                self.config.hidden_size,
+            )
```

## 影响与回归风险
- 影响窗口:正文未说明(正文引用的现行实现基于 v0.3.6.post2)。
- 回归面:diff 仅在 gemma2.py 新增两个方法、未改既有代码路径;直接影响 gemma2 的 LoRA 权重加载维度计算,其余模型走各自实现不受影响。

## 关联
- llama3 config: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/config.json
- gemma2 config: https://huggingface.co/google/gemma-2-2b-it/blob/main/config.json
- 当时 `get_hidden_dim` 实现: https://github.com/sgl-project/sglang/blob/v0.3.6.post2/python/sglang/srt/models/llama.py#L326-L339
- 被绕过的检查: https://github.com/sgl-project/sglang/blob/v0.3.6.post2/python/sglang/srt/lora/lora_manager.py#L34-L46

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
