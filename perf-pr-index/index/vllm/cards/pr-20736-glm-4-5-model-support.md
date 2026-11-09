# PR #20736: GLM-4.5 Model Support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zRzRzRzRzRzRzR |
| 合并 | 2025-07-19 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, performance, new-model, frontend, speculative-decoding, ready, tool-calling |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含约 12 个文件(7 个超长文件被省略,含模型实现) |
| 链接 | https://github.com/vllm-project/vllm/pull/20736 |

## 一句话结论

为 vLLM 接入 THUDM/GLM-4.5 MoE 模型(Glm4MoeForCausalLM),含 MTP 投机解码与 glm4_moe 工具调用解析器。

## 动机

PR 正文仅一句话("GLM-4 Update of Moe Support"),动机无法从正文引用;从标题与 diff 可确认目的:让 `THUDM/GLM-4.5`(`Glm4MoeForCausalLM`,T+I+V 多模态)在 vLLM 上可推理。该模型是 MoE 架构并自带 MTP(nextn)层,需要新增模型实现、MTP 投机解码接线与配套的 XML 风格工具调用解析器,现有 GLM4V/GLM-4.1V 代码不覆盖。

## 优化点

新增 `Glm4MoeForCausalLM` 模型实现(文件在省略的 diff 中,按 supported_models/registry 接线可确认);MoE 结构复用 DeepseekV2/V3 的新 Moe 支持("GLM-4 Update of Moe Support")。投机解码:配置层新增 `glm4_moe_mtp` model_type,`hf_config_override` 把 `Glm4MoeForCausalLM` 改写为 `Glm4MoeMTPModel`(num_hidden_layers 置 0、带 num_nextn_predict_layers),draft 与 target 共享同一 checkpoint,方法归入 `deepseek_mtp` 族(单层 MTP,超过 1 个投机 token 打 warning)。工具调用:新增 `Glm4MoeModelToolParser`(注册名 `glm4_moe`,基于 deepseekv3 parser 修改),解析 GLM-4.5 的 `<tool_call>` / `<arg_key>` / `<arg_value>` XML 格式为 OpenAI function call,支持流式与非流式、多工具调用、thinking 标签共存与畸形输入容错。基准脚本 `benchmark_moe.py` 与 `benchmark_moe_permute_unpermute.py` 增加该架构条目以便做 MoE kernel 基准。

## 关键改动

```text
vllm/model_executor/models/(glm4_moe 模型实现,diff 被省略)
vllm/config.py: glm4_moe_mtp MTP 方法接线(get_layers_start_end_indices、hf_config_override、SpeculativeConfig)
vllm/entrypoints/openai/tool_parsers/glm4_moe_tool_parser.py: 新增 glm4_moe 工具调用解析器(XML 格式)
tests/tool_use/test_glm4_moe_tool_parser.py: 解析器测试(模块级 skip 状态)
benchmarks/kernels/benchmark_moe.py 等: MoE 基准脚本支持 Glm4MoeForCausalLM 形状推导
```

## 性能声明

PR 正文未提供 benchmark 结果(正文一句话,自动检测无性能数字)。MoE 形状接入 benchmark_moe.py 仅为后续 kernel 基准做准备,PR 未附运行结果。

## 验证与回归风险

测试接线:`tests/models/registry.py` 注册 `Glm4MoeForCausalLM`(THUDM/GLM-4.5,min_transformers_version 4.54,发布时 is_available_online=False)与 `Glm4MoeMTPModel` 投机解码条目;新增 `tests/tool_use/test_glm4_moe_tool_parser.py` 覆盖单/多工具调用、中文参数、thinking 标签、畸形 XML、空参数、流式各分支(注意该测试文件头部带 `pytest.skip(..., allow_module_level=True)`,即合并时处于跳过状态);supported_models.md 文档更新。风险面:MTP 配置改写逻辑(hf_config_override)与 deepseek_mtp/mimo_mtp 共用路径,影响层计数;tool parser 修改自 deepseekv3 parser,流式状态机逻辑较复杂;MoE 实现文件在省略 diff 中无法评估细节。

## 关联

同区域:GLM 系列(GLM4VForCausalLM、Glm4vForConditionalGeneration 已支持)、MTP 同方法族 deepseek_mtp/mimo_mtp、DeepseekV2/V3 MoE 路径(其 MoE 支持被 GLM-4.5 复用)。pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
