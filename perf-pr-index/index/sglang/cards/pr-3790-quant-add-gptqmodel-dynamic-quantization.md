# PR #3790: [QUANT] Add GPTQModel Dynamic Quantization + `lm_head` Quantization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Qubitium |
| 合并 | 2025-03-05 |
| 分类 | Quantization |
| Label | 无 |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含大量文件(约 53 个),另有 48 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/3790 |

## 一句话结论

支持 GPTQ 按 module 的 dynamic 量化配置覆写(bits/group_size 逐层不同、可跳过某层)与量化 `lm_head` 加载,减小模型显存与提升量化自由度。

## 动机

此前所有 GPTQ module 共享同一个 `GPTQConfig`(bits、group_size 等全局一致),无法对不同层用不同量化参数;同时量化过的 `lm_head` 完全无法加载。GPTQModel 已支持 `dynamic` per-module 量化配置,本 PR 把该能力接进 SGLang,使其能加载 GPTQModel 产出的动态量化模型,并对向后兼容所有既有非 dynamic GPTQ 模型。dynamic 覆写逻辑代码直接取自 vLLM 0.7.3(同为作者所写)。

## 优化点

新增 SGLang 本地的 `GPTQConfig` / `GPTQMarlinConfig` / `MarlinConfig`(`quantization/gptq.py`),从 `quantize_config.json` 解析 `dynamic`(regex 规则字典)、`lm_head`(bool)字段。`dynamic` 匹配逻辑:规则以 module 全路径 prefix(如 `mlp.down_proj`)做 regex 匹配;`-:` 前缀为 negative match,命中模块整体跳过量化、按非量化加载(`UnquantizedLinearMethod` / `UnquantizedEmbeddingMethod`);`+:` 前缀为 positive match,按规则 dict 覆写该模块的 bits / group_size / desc_act / sym 等属性;无命中则用基础 `GPTQConfig`。实现上 `get_linear_quant_method` 对 config 做 `deepcopy` 后按 prefix 调 `override_config`(重算 pack_factor,gptq_marlin 重解析 quant_type 并校验 TYPE_MAP,gptq 校验 2/3/4/8 bit);`prefix` 通过 `add_prefix` 一路传进各模型层的 linear/attention 创建,使 dynamic 覆写在 GPTQ linear layer 创建前生效(改动覆盖 baichuan、chatglm 等大量模型文件的 prefix 传递)。`lm_head` 侧:`lm_head.linear_method` 属性统一改为 `lm_head.quant_method`,`ParallelLMHead` 在 `lm_head_quantized=True` 时返回量化 method,使量化 lm_head 可加载;GGUF logits 路径同步改用 `quant_method.apply`。

## 关键改动

```text
python/sglang/srt/layers/quantization/gptq.py: 新增本地 GPTQConfig/GPTQMarlinConfig/MarlinConfig(解析 dynamic、lm_head 字段)
python/sglang/srt/layers/quantization/__init__.py: override_config/get_dynamic_override/get_linear_quant_method + gptq_get_quant_method 分派 GPTQ/GPTQMarlin
python/sglang/srt/layers/vocab_parallel_embedding.py: linear_method -> quant_method 改名,ParallelLMHead 支持量化 method
python/sglang/srt/layers/logits_processor.py: lm_head.linear_method.apply -> lm_head.quant_method.apply
docs/backend/quantization.md: 增加 GPTQModel 离线量化示例与文档
```

## 性能声明

PR 未注明 benchmark 结果(checklist 中吞吐/延迟 benchmark 与精度评估两项未勾选)。性能收益为间接性质:dynamic 允许对敏感层跳过量化或用更高 bits、对其他层用更低 bits,量化 `lm_head` 减少该矩阵的显存占用;具体数字 PR 未提供。

## 验证与回归风险

TODO 中勾选"Validate Working State"与"Add `dynamic` GPTQ model unit tests"(diff 被截断部分可能含单测文件)。风险面:`linear_method -> quant_method` 是全局改名,影响 embedding/lm_head/logits 等多条路径;`override_config` 在 deepcopy 上重算 pack_factor(整数除法 `32 // weight_bits`,与 `Fraction(32, weight_bits)` 的基础初始化并存);gptq_get_quant_method 对 `LinearBase` 的判断从直接返回 MarlinLinearMethod 改为走 dynamic 分派,非 dynamic 模型行为依赖 negative/no-match 路径正确;`print("quant_method", ...)` 遗留在 vocab_parallel_embedding 中。

## 关联

同分类:GPTQ/AWQ/Marlin 量化栈;上游为 GPTQModel(ModelCloud)与 vLLM 0.7.3 的 dynamic 实现。本卡区域内无直接前序/后续 PR 编号信息,写"无其他已索引关联"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
