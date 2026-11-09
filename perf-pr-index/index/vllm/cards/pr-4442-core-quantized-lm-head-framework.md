# PR #4442: [CORE] Quantized lm-head Framework

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Qubitium |
| 合并 | 2024-07-02 |
| 分类 | Quantization \| tier: major |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 22 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/4442 |

## 一句话结论

新增量化 lm-head 框架，让大词表模型可对未绑定词嵌入的 lm-head 量化，精度损失极小。

## 动机

PR 正文指出 lm-head 可以在精度损失极小（minimal loss in accuracy）的前提下被量化，而 Llama-3 这类模型词表规模非常大（very large vocab sizes），lm-head 权重成为显著开销。此前 vLLM 缺少对 lm-head 的量化支持，AutoGPTQ 集成也无法识别 checkpoint 中已量化的 lm-head。

## 优化点

实现上把 `VocabParallelEmbedding` 重构为用 `create_weights` 创建参数，把 `ParallelLMHead` 重构为通过 `apply()` 生成输出，从而让量化方法接管 logits 计算；把 `quant_config` 一路 plumb 进 `ParallelLMHead` 与全部模型；更新 AutoGPTQ 集成，当 config 中检测到 `lm-head` 时启用量化 lm-head。正文明确仅支持未绑定词嵌入（without tied word embeddings）的模型，并把「支持 fp8 与 int8」「研究量化 embedding」列为 follow-ups。所有模型层改为经 `apply()` 而非直接 `matmul` 计算 logits，LoRA 的 logits processor 也相应改为接收 `lm_head` 对象而非裸权重张量。

## 关键改动

```text
vllm/model_executor/layers/vocab_parallel_embedding.py: 重构 ParallelLMHead，用 apply() 生成输出、create_weights 创建参数
vllm/model_executor/layers/quantization/base_config.py、gptq.py、gptq_marlin.py、marlin.py: 把 quant_config 接入 lm-head，并让 AutoGPTQ 识别量化 lm-head
vllm/model_executor/layers/logits_processor.py、vllm/lora/layers.py: logits 计算改走 linear_method.apply
tests/quantization/test_lm_head.py: 新增测试，验证 GPTQ 量化 lm_head 模型可加载
vllm/model_executor/models/*.py: 把改动 plumb 到各模型（llama、gemma、deepseek 等，diff 保留 20 余个模型文件）
```

```diff
     def _get_logits(
         self,
         hidden_states: torch.Tensor,
-        embedding: torch.Tensor,
+        lm_head: VocabParallelEmbedding,
         embedding_bias: Optional[torch.Tensor] = None,
     ) -> Optional[torch.Tensor]:
         # Get the logits for the next tokens.
-        logits = torch.matmul(hidden_states, embedding.t())
+        logits = lm_head.linear_method.apply(lm_head, hidden_states)
```

## 性能声明

PR 正文未给出吞吐/延迟类定量性能数据，仅给出量化前后精度对比表（由 auto-round 量化、AutoGPTQ 加载测试，测试模型 `LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse`，指标链接指向 Meta-Llama-3-8B-Instruct 的评测，配置 BF16 / w4g128 w/o lm-head / w4g128 with quantized lm-head）。原文数字：Avg. 0.6352 / 0.6312 / 0.6303；mmlu 0.6386 / 0.6306 / 0.6318；winogrande 0.7143 / 0.7238 / 0.7269；truthfulqa_mc1 0.3623 / 0.3537 / 0.3525；rte 0.6751 / 0.6859 / 0.6679；piqa 0.7867 / 0.7797 / 0.7802；openbookqa 0.3400 / 0.3300 / 0.3320；lambada_openai 0.7182 / 0.7200 / 0.7173；hellaswag 0.5769 / 0.5699 / 0.5701；boolq 0.8297 / 0.8309 / 0.8284；arc_easy 0.8152 / 0.8089 / 0.8106；arc_challenge 0.5299 / 0.5102 / 0.5154。硬件与推理 workload 条件 PR 未注明测试条件。

## 验证与回归风险

正文说明「added tests for ~all models that can fit in our L4 CI to confirm this does not break anything」；新增 `tests/quantization/test_lm_head.py`，以 `MODELS_QUANT` 参数化验证量化 lm_head 模型的 `linear_method` 为 `GPTQLinearMethod` / `GPTQMarlinLinearMethod` / `MarlinLinearMethod`，未量化模型为 `UnquantizedLinearMethod`，并跑 generate_greedy；`tests/lora/test_layers.py` 同步改为传入 `lm_head`。回归风险：改动 plumb 到全部模型的 lm-head 与 logits 计算路径，并触及 LoRA logits processor；绑定词嵌入的模型按正文不在支持范围，PR 未注明更多受影响路径或硬件。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
