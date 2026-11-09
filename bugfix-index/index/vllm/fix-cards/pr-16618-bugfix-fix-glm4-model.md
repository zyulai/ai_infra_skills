# PR #16618: [Bugfix] Fix GLM4 model

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#16618](https://github.com/vllm-project/vllm/pull/16618) |
| 作者 | intervitens |
| 合并日期 | 2025-04-17 |
| labels | documentation, ready |
| 热度 | 👍 7 · 💬 34 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
GLM4 模型完全无法加载运行;修正 rotary 维度/风格与 residual-layernorm 顺序,使输出基本对齐 HF transformers。

## 问题现象
正文描述:GLM4 模型目前完全不能工作,加载即失败(fails to load at all)。本 PR 使模型可以加载,并让输出与 HF transformers 的输出基本一致(mostly identical)。修复 #16617、#16655、#16687、#16740。

## 根因
正文未披露根因,仅说明模型无法加载。diff 改动所暗示的机制(由 diff 推断):`rotary_dim` 按 `partial_rotary_factor`(默认 0.5)缩放,而 GLM4 应为全 head_dim 旋转且非 Neox 风格;`forward` 中在 layernorm 之外额外执行 `hidden_states = residual + hidden_states`,与 `post_attention_layernorm(hidden_states, residual)` 的调用叠加,残差流被重复累加。

## 修复方案
`vllm/model_executor/models/glm4.py`:
- `self.rotary_dim = self.head_dim`(不再乘 `partial_rotary_factor`);
- rotary embedding 增加 `is_neox_style=False`;
- `forward` 删除 attention 与 MLP 后的两处 `hidden_states = residual + hidden_states`,改为接收 `post_attention_layernorm(hidden_states, residual)` 返回的 `(hidden_states, residual)` 二元组。

另将 `docs/source/models/supported_models.md` 与 `tests/models/registry.py` 的示例模型从 `THUDM/GLM-4-32B-Chat-0414` 改为 `THUDM/GLM-4-32B-0414`。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:仅 `Glm4ForCausalLM`(glm4.py)模型路径及其测试注册/文档条目;输出目标为「与 HF transformers 基本一致」,正文未附量化对比数据。

## 关联
- Fixes [#16617](https://github.com/vllm-project/vllm/issues/16617)
- Fixes [#16655](https://github.com/vllm-project/vllm/issues/16655)
- Fixes [#16687](https://github.com/vllm-project/vllm/issues/16687)
- Fixes [#16740](https://github.com/vllm-project/vllm/issues/16740)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
