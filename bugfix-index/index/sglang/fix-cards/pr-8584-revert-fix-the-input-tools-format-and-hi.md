# PR #8584: Revert "Fix the input tools format and history tool_calls in OpenAI API  (#6556)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8584](https://github.com/sgl-project/sglang/pull/8584) |
| 作者 | CatherineSue |
| 合并日期 | 2025-07-30 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论
回滚 #6556:其为 Kimi chat template 的 workaround 但破坏其他模型,恢复 tools 的 `function.model_dump()` 并删除历史 tool_calls 的 json 反序列化处理。

## 问题现象
其他模型被 #6556(Fix the input tools format and history tool_calls in OpenAI API)破坏,本 PR 以 `Fixes: #8580` 关联对应 issue。正文未描述具体报错;并说明 #6556 看起来只是针对 Kimi chat template 的 workaround("The original PR #6556 seems to be a work-around for Kimi's chat template. But it breaks other models.")。

## 根因
正文披露:#6556 是针对 Kimi chat template 的 workaround,会破坏其他模型。从 diff 看,#6556 的两处改动(机制描述由 diff 推断)对非 Kimi 模型产生了副作用:(1) tools 序列化从 `item.function.model_dump()` 改为 `item.model_dump()`,把 OpenAI 的外层 Tool 对象(含 `type`/`function` 包装)直接下发,改变了下游拿到的 tools 结构;(2) 在 `_apply_jinja_template` 中对历史消息 `tool_calls[].function.arguments` 做 `json.loads` 反序列化,改变了进入 chat template 的历史消息形态。

## 修复方案
`python/sglang/srt/entrypoints/openai/serving_chat.py`:

- `_process_messages` 中 tools 恢复为按 `item.function.model_dump()` 序列化:

```diff
-                    tools = [item.model_dump() for item in request.tools]
+                    tools = [item.function.model_dump() for item in request.tools]
```

(tool_choice 指定函数的分支同样恢复)

- 删除 `_apply_jinja_template` 中对历史 `tool_calls` 的 `arguments` 做 `json.loads` 的整段处理(#6556 新增的 ~19 行)。

## 影响与回归风险
- 影响窗口:#6556 合并至本回滚之间。
- 回归面:OpenAI 兼容 chat API 的 tool 解析路径——tools 入参与历史 tool_calls 的处理回到 #6556 之前;Kimi 场景若依赖该 workaround,其行为回归(正文未评价);#8580 所报问题是否需另行修复,正文未说明。

## 关联
- [#8580](https://github.com/sgl-project/sglang/issues/8580)(本 PR 修复的 issue)
- [#6556](https://github.com/sgl-project/sglang/pull/6556)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
