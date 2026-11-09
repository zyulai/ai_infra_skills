# PR #32542: Enable Eagle3 speculative decoding for Pixtral (LlavaForConditionalGeneration)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gopalsarda |
| 合并 | 2026-01-21 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, performance, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, cpu, kv-connector, nvidia |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用(单文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/32542 |

## 一句话结论

让 Pixtral(LlavaForConditionalGeneration)实现 Eagle3 所需的 aux hidden state 接口,支持投机解码。

## 动机

vLLM 的 Eagle3 投机解码要求目标模型实现 SupportsEagle3 接口并提供中层 aux hidden state 供 drafter 消费;Pixtral 走 LlavaForConditionalGeneration 类,此前未实现该接口,无法配合 Eagle3 使用。

## 优化点

给 LlavaForConditionalGeneration 挂上 SupportsEagle3,并实现两个接口方法:set_aux_hidden_state_layers 把层号写入 language model 的 aux_hidden_state_layers;get_eagle3_aux_hidden_state_layers 返回 Eagle3 约定的三个中层索引 (2, num_layers // 2, num_layers - 3)。改动只有接口适配,不触碰模型前向或 kernel。

## 关键改动

```text
vllm/model_executor/models/llava.py: 类声明加 SupportsEagle3;新增两个 Eagle3 接口方法
```

```python
def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
    num_layers = len(self.get_language_model().model.layers)
    return (2, num_layers // 2, num_layers - 3)
```

## 性能声明

PR 未给出吞吐/延迟数字:正文仅说明"用本地训练的 speculator 测试,观察到 reasonable acceptance rates",未附具体 acceptance rate 或加速比。性能收益为使 Eagle3 投机解码在 Pixtral 上可用这一能力本身。

## 验证与回归风险

验证为作者本地 speculator 的接受率观察,PR 未列 CI 测试命令或数字。风险很小:纯接口注册;风险点在于 aux hidden state 层选择 (2, L/2, L-3) 是否与该 drafter 训练时的取层一致(取决于用户训练的 speculator),以及多模态路径下语言模型 hidden state 的抽取时机。

## 关联

同分类 Speculative Decoding 卡:#34457(sparse MLA+MTP full cudagraph)、#59082(CPU MTP 调度)、#63873(bench 投机状态)。Eagle3 基础支持见仓库已有卡 pr-16937(v1 spec decode Eagle3 support)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
