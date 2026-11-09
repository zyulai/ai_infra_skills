# PR #21808: [Bugfix] Fix shape mismatch assertion error when loading Gemma3n model with BitsAndBytes quantization

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#21808](https://github.com/vllm-project/vllm/pull/21808) |
| 作者 | sydarb |
| 合并日期 | 2025-07-30 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
Gemma3n + BitsAndBytes 加载即断言失败:若干线性层漏传 `quant_config`,补齐后模型可正常加载与生成。

## 问题现象
用 BitsAndBytes 量化加载 Gemma3n(`unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit`,`quantization=bitsandbytes`,`load_format=bitsandbytes`,vLLM v0.10.1.dev125+g2cc571199,`enforce_eager=True`,`max_model_len=4000`)时,权重加载第二个 shard 阶段直接崩溃:
```
File ".../models/gemma3n.py", line 727, in load_weights
    weight_loader(param, loaded_weight)
File ".../layers/linear.py", line 570, in weight_loader
    assert param_data.shape == loaded_weight.shape
AssertionError
```
即 shape mismatch 断言错误,引擎无法启动。影响面:BitsAndBytes 量化下的 Gemma3n(Gemma3nForConditionalGeneration)加载。修复后同脚本正常完成两遍 shard 加载,Model loading took 7.5565 GiB,并成功生成文本("I'm doing well, thank you for asking! ...")。

## 根因
正文只写「Fix #21743」,未直接披露根因分析。
(由 diff 推断)Gemma3n 中多处线性层在构造时没有把 `quant_config` 传下去:`Gemma3nLaurelBlock` 的 `linear_left`/`linear_right`、`per_layer_input_gate`、`per_layer_projection`、`per_layer_model_projection`、`altup_projections`、`altup_unembed_projections`。BitsAndBytes 加载流程按层是否携带 quant_config 决定是否走 bnb 量化参数路径;这些漏传的层保持普通(未量化)参数形状,而 bitsandbytes_loader 的权重迭代器按量化流程提供权重,形状对不上,`ColumnParallelLinear.weight_loader` 中 `assert param_data.shape == loaded_weight.shape` 抛出 AssertionError。为这些层补传 `quant_config` 后,它们被正确纳入量化路径,加载恢复正常。

## 修复方案
改 `vllm/model_executor/models/gemma3n.py`,仅做一件事:给上述所有缺失的线性层补 `quant_config=quant_config` 并沿调用链传入:
- `Gemma3nLaurelBlock.__init__` 增加 keyword-only 参数 `quant_config`,其 `linear_left`(ColumnParallelLinear)与 `linear_right`(RowParallelLinear)都加 `quant_config=quant_config`;`Gemma3nTextModel` 构造 `Gemma3nLaurelBlock` 时传入 `quant_config`。
- `per_layer_input_gate`、`per_layer_projection`(Gemma3nTextModel)与 `per_layer_model_projection`、`altup_projections`、`altup_unembed_projections`(Gemma3nModel)各补 `quant_config=quant_config`。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围(问题对应 #21743)。
- 回归面:仅 Gemma3n 模型的层构造参数:这些层在非量化路径下 `quant_config=None` 行为不变;在 BitsAndBytes(及其他依赖 quant_config 的量化方式)路径下,原先未被量化的 laurel/per-layer/altup 投影层现在会被量化替换——这也意味着这些层从此参与量化,显存占用与数值行为随量化方式变化(正文 After 日志显示加载成功,未做精度评估)。

## 关联
- [Fix #21743](https://github.com/vllm-project/vllm/issues/21743)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
