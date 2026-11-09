# PR #5684: [Bugfix] Fix sampling_params passed incorrectly in Phi3v example

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5684](https://github.com/vllm-project/vllm/pull/5684) |
| 作者 | Isotr0py |
| 合并日期 | 2024-06-19 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
Phi3v 示例把 `sampling_params` 塞进 prompt 字典传给 `llm.generate` 而非作为关键字参数,采样参数不生效,可能产生混乱/不完整的回答;改为关键字参数传入并移除硬编码的 `max_model_len=4096`。

## 问题现象
按 `examples/phi3v_example.py` 示例使用 Phi-3 vision 模型时,定义好的 `SamplingParams(temperature=0, max_tokens=64)` 实际未按预期生效。正文描述该错误传参 "may create confusing/incompleted answer",即可能导致模型给出令人困惑或不完整的回答。正文未提供更具体的复现日志。

## 根因
PR 正文披露:示例文件中 `sampling_param` 被错误传入("Fix the incorrectly passed `sampling_param` in `examples/phi3v_example.py`")。结合 diff 可见具体机制(由 diff 推断):`sampling_params` 作为字典的一个条目与 `prompt`、`multi_modal_data` 一起传给 `llm.generate({...})`,而不是作为 `llm.generate(..., sampling_params=...)` 的独立关键字参数,引擎无法把字典里的该条目识别为采样参数,于是其 temperature/max_tokens 设置被忽略。

## 修复方案
修改 `examples/phi3v_example.py`:

1. 把 `sampling_params` 从传入 `llm.generate` 的字典中移出,改为方法的关键字参数;
2. 删除 `LLM(...)` 中的 `max_model_len=4096` 一行。

关键 diff 摘录:

```diff
-    outputs = llm.generate({
-        "prompt": prompt,
-        "sampling_params": sampling_params,
-        "multi_modal_data": ImagePixelData(image),
-    })
+    outputs = llm.generate(
+        {
+            "prompt": prompt,
+            "multi_modal_data": ImagePixelData(image),
+        },
+        sampling_params=sampling_params)
```

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:改动仅涉及 `examples/phi3v_example.py` 这一个示例文件,不影响库代码与任何生产路径。示例的行为恢复为 temperature=0、max_tokens=64 真正生效;删除 `max_model_len=4096` 后,示例改用模型默认的最大长度配置(按 diff 如实陈述,正文未解释删除原因)。

## 关联
- #4986(正文提到 "Related to a issue mentioned in #4986"): https://github.com/vllm-project/vllm/issues/4986

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
