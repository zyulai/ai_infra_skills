# PR #3016: Fix: Output text is always truncated in some models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3016](https://github.com/vllm-project/vllm/pull/3016) |
| 作者 | HyperdriveHustle |
| 合并日期 | 2024-03-01 |
| labels | (none) |
| 热度 | 👍 8 · 💬 3 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
stop_token_ids 触发停止时输出被无条件砍掉 stop_str 长度的文本;改为仅在输出确以 stop_str 结尾时截断。

## 问题现象
vllm==0.3.2 用 Yi-34B-Chat 推理时,`CompletionOutput.text` 总是被截断(正文给出截断示例:文本止于"……我会尽力帮助你。有什",finish_reason=stop)。触发条件:在 `sampling_params` 中配置了 `stop_token_ids`。

## 根因
正文:修复前当 `sampling_params` 配置了 `stop_token_ids` 时,引擎在截断序列前没有先检查 `stop_str` 是否真的存在于输出末尾("This PR ensures that when stop_token_ids is configured in the sampling_params, it first checks for the existence of the stop_str before truncating the sequence")。从 diff 看旧逻辑是:只要 `include_stop_str_in_output` 为假且 `stop_string` 非空,就无条件砍掉输出末尾 `len(stop_string)` 个字符——停止由 token id 触发而文本并不以 stop_str 结尾时,砍掉的是真实内容(由 diff 推断)。

## 修复方案
改 `vllm/engine/llm_engine.py` 的 `_finalize_sequence`:`include_stop_str_in_output` 时直接返回;仅当 `stop_string` 非空且 `seq.output_text.endswith(stop_string)` 时才执行截断。

```diff
-        if not sampling_params.include_stop_str_in_output and stop_string:
+        if sampling_params.include_stop_str_in_output:
+            return
+
+        if stop_string and seq.output_text.endswith(stop_string):
             # Truncate the output text so that the stop string is
             # not included in the output.
             seq.output_text = seq.output_text[:-len(stop_string)]
```

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:所有经 `_finalize_sequence` 的停止字符串截断逻辑(0.x 引擎路径);`include_stop_str_in_output=True` 行为不变(提前返回等价于原"不截断"分支);由 stop 字符串命中的截断行为不变,由 stop token id 命中时不再误删尾部文本。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
