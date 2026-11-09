# PR #57563: [fix] Mistral-Large-3 accuracy regression on `main`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57563](https://github.com/vllm-project/vllm/pull/57563) |
| 作者 | jdebache |
| 合并日期 | 2026-09-18 |
| labels | ready, deepseek, mistral, DSv4, kimi, k3, DSv4.1 |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Mistral-Large-3 在 main 上 GSM8K 掉约 4.6 分:YaRN 无 mscale 时被误走 deepseek_yarn;改为按 `attention_factor == 1.0` 选 `deepseek_llama_scaling` 修复。

## 问题现象
Mistral-Large-3 在 `main` 上出现精度回归:GSM8K 五次评测均值 88.05%(sd 0.97),而 #56446 合并前(2026-09-10)测得 92.2 / 92.8%。影响面:走 Mistral config remap 的 Mistral-Large-3(正文强调它是唯一既走 Mistral remap 又要求无 magnitude 缩放 YaRN 的模型,其他模型未受影响)。修复后同条件 GSM8K 均值 92.63%(sd 0.30),+4.58 分(每轮约多答对 60 题,Welch t=10.1)。测试配置:TP1 / DP4 / EP4,`/models/Mistral-Large-3-675B-Instruct-2512-NVFP4`,`--max-model-len 5120 --tokenizer-mode mistral --language-model-only --seed 42` 等,1319 题 5-shot temperature 0。

## 根因
正文披露:Mistral-Large-3 的 Mistral 格式 `params.json` 带 `"yarn": {"factor": 36, ..., "apply_scale": false}` 与 `llama_4_scaling`,即「YaRN 频率插值但**不做** YaRN magnitude 校正(mscale)」外加 llama4 式位置相关 attention 缩放。vLLM 将其作为 `MistralLarge3ForCausalLM`(`DeepseekV3ForCausalLM` 子类)服务,DeepSeek attention 代码运行在 Mistral adapter(`vllm/transformers_utils/configs/mistral.py`)合成的 config 上。提交 `c191787a68`(「[Bugfix] Align vLLM YaRN with Transformers and stop re-scaling max_model_len」,#56446,2026-09-11 合并)之后,通过 `get_config(..., config_format="mistral")` 加载该 checkpoint 得到的 `rope_parameters` 为 `attention_factor: 1.0` 且没有 `apply_yarn_scaling` 键;而模型代码用 `rope_parameters.get("apply_yarn_scaling", True)` 判断 rope 类型,键缺失即默认 True → 选择 `deepseek_yarn`,对该模型施加了多余的 `yarn_get_mscale(factor)^2` attention 缩放(测试 docstring 中对该机制有明确说明),导致精度下降。

## 修复方案
统一翻转 rope 类型判定条件:从「`apply_yarn_scaling`(缺省 True)」改为「`attention_factor == 1.0` → `deepseek_llama_scaling`,否则 `deepseek_yarn`」:
```python
config.rope_parameters["rope_type"] = (
    "deepseek_llama_scaling"
    if config.rope_parameters.get("attention_factor") == 1.0
    else "deepseek_yarn"
)
```
共改 8 处相同逻辑:`vllm/model_executor/models/AXK1.py`(2 处)、`vllm/model_executor/models/deepseek_v2.py`(2 处)、`vllm/models/deepseek_v32/attention.py`、`vllm/models/deepseek_v4/common/rope.py`、`vllm/models/deepseek_v41/common/rope.py`、`vllm/models/glm5next/common/attention.py`、`vllm/models/kimi_k3/nvidia/mla.py`;并在 `tests/transformers_utils/test_config.py` 新增测试,断言 `yarn.apply_scale: false` 经 `adapt_config_dict` 后 `rope_parameters["attention_factor"] == 1.0`。

## 影响与回归风险
- 影响窗口:#56446(commit `c191787a68`,2026-09-11 合并)之后至本 PR 之前的 `main` 提交,Mistral-Large-3 精度回归约 -4.6 分(正文给出对比数据)。
- 回归面:所有经上述 8 处代码选择 DeepSeek 式 rope 类型的模型家族(deepseek_v2/v3/v3.2/v4/v4.1、GLM5Next、Kimi K3、AXK1 及以 MistralLarge3ForCausalLM 服务 Mistral config 的模型):`attention_factor == 1.0` 的 config 现在会被判为 `deepseek_llama_scaling` 而非 `deepseek_yarn`;此前依赖「缺省 apply_yarn_scaling=True→deepseek_yarn」行为的真实 YaRN config 若未携带 `attention_factor` 键,`.get("attention_factor")` 返回 None ≠ 1.0,仍走 `deepseek_yarn`,与旧默认一致。

## 关联
- [#56446](https://github.com/vllm-project/vllm/pull/56446)(回归引入 PR,commit `c191787a68`,正文提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
