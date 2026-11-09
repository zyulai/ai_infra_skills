# PR #24613: [Bug] [Spec Decode] Fix model_initialization test and mismatch in aux_hidden_layers

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24613](https://github.com/vllm-project/vllm/pull/24613) |
| 作者 | wwl2755 |
| 合并日期 | 2025-09-11 |
| labels | ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
初始化测试误用 speculative_model 且 dummy 配置把 num_layers 置 1,致 EAGLE3 aux 层索引为 (2,0,-2) 触发形状不匹配;改为用目标模型并保留原始层数。

## 问题现象
`tests/models/test_initialization.py` 中 EAGLE/EAGLE3 相关模型(如 `EagleLlamaForCausalLM`、`Eagle3LlamaForCausalLM`、`LlamaForCausalLMEagle3`)初始化测试失败:先是用错了模型导致 model_initialization 出错;改用正确目标模型后又因 `hidden_states + residual` 形状不同而报错。日志显示 `self.aux_hidden_state_layers: (2, 0, -2)`,而期望值应为 `(2, 20, 37)`。影响面为测试环境的这几个模型初始化路径。

## 根因
两个问题(均来自正文):
1. model_initialization 测试应使用与 `speculative_model` 不同的 target model 来初始化,此前用错了模型。
2. 改用正确目标模型后,测试里 `dummy_hf_overrides` 把 `num_layers` 初始化为 1(正文引用 tests/models/utils.py v0.10.2rc1 L387),而 aux_hidden_layer 由 `(2, num_layers // 2, num_layers - 3)` 决定,于是得到 `(2, 0, -2)`:layer-0 被当作 aux 层,走进了 `hidden_states + residual` 代码路径(此时 hidden_states 是张量而 residual 为 None),引发形状不匹配。

## 修复方案
- `tests/models/registry.py`:`_HfExamplesInfo` 新增 `use_original_num_layers` 标志;`EagleLlamaForCausalLM`/`Eagle3LlamaForCausalLM`/`LlamaForCausalLMEagle3` 的主模型改为真实目标模型(meta-llama/Meta-Llama-3-8B-Instruct、meta-llama/Llama-3.1-8B-Instruct、Qwen/Qwen3-8B),EAGLE3 条目设置 `use_original_num_layers=True`。
- `tests/models/test_initialization.py`:把该标志透传给 `dummy_hf_overrides`。
- `tests/models/utils.py`:`dummy_hf_overrides` 在 `use_original_num_layers=True` 时保留 config 原始层数,否则维持最小层数:

```python
if use_original_num_layers:
    num_layers = getattr(text_config, 'num_layers', 1)
    num_hidden_layers = getattr(text_config, 'num_hidden_layers', 1)
else:
    num_layers = 1
    num_hidden_layers = (3 if model_arch
                         == "Gemma3nForConditionalGeneration" else 1)
```

正文称相关测试与 `examples/offline_inference/spec_decode.py --method eagle3` 离线推理全部通过。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:改动全部位于 `tests/models/` 下的测试辅助代码(registry、test_initialization、utils),不影响 vLLM 运行时;使用 `use_original_num_layers=True` 的测试将加载完整层数的 dummy 配置,可能增加这几个用例的资源消耗(由 diff 推断)。

## 关联
- https://github.com/vllm-project/vllm/blob/v0.10.2rc1/tests/models/utils.py#L387

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
