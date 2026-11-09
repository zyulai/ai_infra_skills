# PR #26873: [Bugfix][Multi Modal] Fix incorrect Molmo token processing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26873](https://github.com/vllm-project/vllm/pull/26873) |
| 作者 | sangho-vision |
| 合并日期 | 2025-10-15 |
| labels | ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Molmo 在线 chat 服务聊天模板被套用两次,prompt 出现双重 User:/Assistant:;get_tokens_input 改用 message_format="none"。

## 问题现象
在线以 chat completion 方式服务 Molmo 模型时,prompt 中聊天模板被应用了两次:输出 decode 后为 `User: User: Describe the image. Assistant: Assistant:`,即 "User:" 与 "Assistant:" 各出现两遍。触发路径:vLLM 先对输入文本应用 Molmo 的 chat template 并 tokenize,随后调用 `MolmoMultiModalProcessor` 的自定义 `_apply_hf_processor_tokens_only` 做后续处理,而该方法内部又调用了 Molmo HF processor 的 `get_tokens_input`,再次应用 chat template,造成 double templating。正文提供了基于 `allenai/Molmo-7B-D-0924`、`LLM.chat` 的完整复现脚本;修复后同一脚本输出为 `User: Describe the image. Assistant:`。

## 根因
正文披露:`_apply_hf_processor_tokens_only` 内部调用 `get_tokens_input` 时使用模型默认配置的 `message_format`,这会再次套用 chat template;而此时外层已经套用过一次,导致双重模板化。

## 修复方案
改动 `vllm/model_executor/models/molmo.py` 的 `_apply_hf_processor_tokens_only`:调用 `get_tokens_input` 时 `message_format` 从 `processor.message_format` 硬编码为 `"none"`,避免二次应用模板;并更新注释说明 chat template 已应用、需按 `always_start_with_space` 决定是否前置空格。关键 diff 摘录:

```python
# The chat template is already applied to the prompt tokens
# Use message_format="none" to avoid applying it again
tokens = processor.processor.get_tokens_input(  # type: ignore
    self.info.get_tokenizer().decode(prompt_tokens),
    message_format="none",
    always_start_with_space=processor.always_start_with_space,
)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:影响 Molmo 走 `_apply_hf_processor_tokens_only`(prompt tokens 已套模板)这条输入路径的 token 构造;`message_format` 不再跟随 processor 默认配置,若该 processor 在其他场景依赖默认 `message_format` 生成不同格式,此函数行为与其不再一致(按 diff,此函数仅在模板已应用的前提下被调用)。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
