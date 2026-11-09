# PR #9114: [Bugfix] Fix accuracy-test-1-gpu failure caused by `builtin_tools`

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9114](https://github.com/sgl-project/sglang/pull/9114) |
| 作者 | CatherineSue |
| 合并日期 | 2025-08-12 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

#9043 给 chat template 传入多余的 `builtin_tools=[]`,与 Llama3.1 模板的 `tools` 冲突致精度测试失败;删除该参数修复。

## 问题现象

单卡精度测试 accuracy-test-1-gpu 失败。受影响的是使用 Llama-3.1-8B-Instruct 等模型的评测:其 chat template(见 meta-llama/Llama-3.1-8B-Instruct 的 default 模板 L24-L29)定义了 `builtin_tools` 字段,该字段与 `tools` 存在冲突,导致渲染出的对话模板不正确、模型输出异常,精度测试失败。

## 根因

按 PR 正文:#9043 为支持 gpt-oss,在调用 `tokenizer.apply_chat_template` 时显式传入了 `builtin_tools=[]`。但该字段与 `tools` 功能冲突;而且对 gpt-oss 本身也不需要设置——模板会检查 `builtin_tools` 是否被设置,只要不传即可。这个多余参数对所有走 `_apply_jinja_template` 的模型生效,于是破坏了 Llama3.1 的精度测试。

## 修复方案

改 `python/sglang/srt/entrypoints/openai/serving_chat.py` 的 `_apply_jinja_template`,在两处 `apply_chat_template` 调用中删除 `builtin_tools=[]`:

```diff
                 add_generation_prompt=True,
                 tools=tools,
                 reasoning_effort=request.reasoning_effort,
-                builtin_tools=[],
                 **(
                     request.chat_template_kwargs if request.chat_template_kwargs else {}
                 ),
```

按正文,这不影响 gpt-oss 模型:模板检测的是 `builtin_tools` 是否被设置,不传即为原预期行为。

## 影响与回归风险

- 影响窗口:#9043 合入之后、本修复之前的版本。
- 回归面:OpenAI 兼容 serving 的 chat completions 模板渲染路径(`_apply_jinja_template` 两个分支)。删除的是公共参数,所有依赖默认模板行为的模型恢复原状;gpt-oss 仍按正文说明不受影响。作者附上 `test_eval_accuray_large` 的 4 张截图作为验证。

## 关联

- #9043(引入 `builtin_tools=[]` 的 PR)
- Llama-3.1-8B-Instruct chat template(L24-L29)引用:https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct?chat_template=default#L24-L29

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
