# PR #24686: Fix implementation divergence for BLOOM models between vLLM and HuggingFace when using prompt embeds

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24686](https://github.com/vllm-project/vllm/pull/24686) |
| 作者 | qthequartermasterman |
| 合并日期 | 2025-09-12 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
vLLM 的 Bloom `get_input_embeddings` 预先套了 word_embeddings_layernorm,与 HF 假设不一致,传入 prompt embeds 时前向结果发散;把 layernorm 移入 forward 统一应用即可。

## 问题现象
使用 prompt embeds(`inputs_embeds`)时,BLOOM 模型在 vLLM 与 HuggingFace 之间的实现出现分歧:`PretrainedModel.get_input_embeddings()(token_ids)` 与 vLLM 的 `BloomModel.get_input_embeddings(token_ids)` 会给出不同的 embeddings,这种分歧在传入 prompt embeds 时导致模型 forward 的下游结果不同。此前未被测出的原因是:`bigscience/bloom-560m` 的 prompt embeds `common_models.py::test_models` 测试只在 v1 引擎下运行,而 prompt embeds 输入只在 v0 引擎激活时才被测试。

## 根因
正文披露:HF 的 Bloom 实现假设 `inputs_embeds` **没有**预先套上 word embedding 的 layernorm;而 vLLM 的 `BloomModel.get_input_embeddings` 返回的是 `word_embeddings_layernorm(word_embeddings(input_ids))`,即已套过 layernorm,两边语义不一致。正文引用了 HF 源码 modeling_bloom.py L545-L562 作为对照。

## 修复方案
改 `vllm/model_executor/models/bloom.py`:
1. `get_input_embeddings` 不再套 layernorm,直接返回 `self.word_embeddings(input_ids)`;
2. `forward` 中无论 hidden_states 来自 `inputs_embeds` 还是 `input_ids`,统一在进入 decoder 层之前应用 `self.word_embeddings_layernorm(hidden_states)`。

```diff
     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
-        return self.word_embeddings_layernorm(self.word_embeddings(input_ids))
+        return self.word_embeddings(input_ids)
...
                 hidden_states = inputs_embeds
             else:
                 hidden_states = self.get_input_embeddings(input_ids)
+            hidden_states = self.word_embeddings_layernorm(hidden_states)
```
同时更新 `tests/models/language/generation/test_common.py`:`test_models` 参数化 `use_prompt_embeds=[True, False]`,删除原先 `use_prompt_embeds = os.getenv("VLLM_USE_V1") == "0"` 的写法。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:Bloom 模型的输入嵌入路径。`get_input_embeddings` 的返回值不再含 layernorm,依赖该方法的调用方(如 prompt embeds 相关 pipeline、pooling/其他复用点)行为改变;forward 内部对两条输入路径统一先套 layernorm,常规 token 输入路径结果应保持不变(layernorm 仍被应用)。

## 关联
- [HF modeling_bloom.py L545-L562](https://github.com/huggingface/transformers/blob/cf084f5b40e19b5a5f946cee75bead6d4247b071/src/transformers/models/bloom/modeling_bloom.py#L545-L562)
- [PR #17615](https://github.com/vllm-project/vllm/pull/17615)
- [PR #24278](https://github.com/vllm-project/vllm/pull/24278)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
