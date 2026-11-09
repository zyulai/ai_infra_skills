# PR #23617: [mypy] Fix incorrect type hint for EAGLE3 support

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23617](https://github.com/vllm-project/vllm/pull/23617) |
| 作者 | DarkLight1337 |
| 合并日期 | 2025-08-26 |
| labels | ready, llama, qwen |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论

EAGLE3 辅助隐藏层标注误写 tuple[int],在 llama/qwen2/qwen3 中改为 tuple[int, ...]。

## 问题现象

正文未描述运行期症状;这是一个纯类型标注(mypy)修复。`tuple[int]` 这一错误标注存在于 llama、qwen2、qwen3 三个模型的 EAGLE3 辅助隐藏状态层相关代码上,会导致 mypy 类型检查不正确。正文无报错日志、无复现描述。

## 根因

按正文披露:类型标注写错了,应使用 `tuple[int, ...]` 而不是 `tuple[int]`,并链接了 PR #22080 下的 review 讨论(r2299697611)作为出处。(由 diff 推断)`tuple[int]` 在类型系统中表示"恰好含一个 int 的元组",而 `aux_hidden_state_layers` 及 `set_/get_eagle3_aux_hidden_state_layers` 实际承载的是可变长度的层索引元组(如 `(2, num_layers // 2, num_layers - 3)`),因此原标注与实际类型不符。

## 修复方案

在三个模型文件中把 EAGLE3 相关的类型标注统一改为变长元组:

- `vllm/model_executor/models/llama.py`:`self.aux_hidden_state_layers: tuple[int] = tuple()` 改为 `self.aux_hidden_state_layers = tuple[int, ...]()`;`set_aux_hidden_state_layers` 参数与 `get_eagle3_aux_hidden_state_layers` 返回值标注改为 `tuple[int, ...]`。
- `vllm/model_executor/models/qwen2.py`:同样的三处修改。
- `vllm/model_executor/models/qwen3.py`:`set_aux_hidden_state_layers` 参数与 `get_eagle3_aux_hidden_state_layers` 返回值标注改为 `tuple[int, ...]`(该文件无初始化行)。

关键 diff 摘录(llama.py):

```diff
-        self.aux_hidden_state_layers: tuple[int] = tuple()
+        self.aux_hidden_state_layers = tuple[int, ...]()
```

运行期逻辑(返回哪几个层)完全未变。

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:仅类型标注,无运行时行为变化;影响 mypy 对这三个模型 EAGLE3 路径的检查结果。若其他模型文件存在同样的 `tuple[int]` 误写,本 PR 未覆盖。纯静态修改,回归风险可忽略。

## 关联

- [PR #22080 讨论](https://github.com/vllm-project/vllm/pull/22080#discussion_r2299697611)(正文链接的 review 出处)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
