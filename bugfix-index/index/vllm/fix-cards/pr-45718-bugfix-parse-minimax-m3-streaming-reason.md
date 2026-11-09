# PR #45718: [Bugfix] Parse MiniMax M3 streaming reasoning by text markers

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45718](https://github.com/vllm-project/vllm/pull/45718) |
| 作者 | Palaiologos1453 |
| 合并日期 | 2026-06-23 |
| labels | bug, ready, tool-calling |
| 热度 | 👍 5 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
MiniMax M3 流式输出中标记文本被拆成多个 token 时 reasoning 漏进 content;改为按文本标记切分提取。

## 问题现象
即使 tokenizer 已把 `<mm:think>` / `</mm:think>` 作为完整词表条目,MiniMax M3 流式生成时 reasoning 标记仍可能以可见标记文本的形式产出。旧的流式解析器只检查单个标记 token ID,被拆分成多个 token 的标记文本会作为 `delta.content` 漏出,reasoning 永远不会填充到 `delta.reasoning`。本 PR 是在 `main` 分支重开此前针对 `m3_release` 的修复(原 PR #45713 因 base 分支在 M3 经 #45381 落地 `main` 后被删除,被 GitHub 自动关闭)。

## 根因
正文披露:旧的流式解析器只依赖单个标记 token ID 做判定,当标记文本在运行时被 token 化为更小的分片时,标记无法被识别,直接透传为 `delta.content`,reasoning 流从未进入 `delta.reasoning`。

## 修复方案
改动集中在 `vllm/reasoning/minimax_m3_reasoning_parser.py`:
- 不再依赖单一标记 token ID,改为围绕 MiniMax M3 标记比较可见文本片段来提取 reasoning/content(新增 `_visible_segments`、`_strip_partial_marker_suffix`、`_visible_delta` 等辅助方法);
- 流式 reasoning-end 状态与文本解析器保持一致,使拆分 token 的结束标记之后即可开始工具解析;
- token-id 辅助方法(`is_reasoning_end`、`is_reasoning_end_streaming`、`extract_content_ids`、`count_reasoning_tokens`)改为识别编码后的标记序列;
- `tests/reasoning/test_minimax_m3_reasoning_parser.py` 新增回归测试,覆盖「词表有标记条目但运行时标记 ID 被拆分」的场景。

```python
+        content_before, _, after_start = text.partition(self.start_token)
+        reasoning, end, content_after = after_start.partition(self.end_token)
+        if end:
+            return reasoning or None, (content_before + content_after) or None
+
+        reasoning = self._strip_partial_marker_suffix(reasoning, self.end_token)
+        return reasoning or None, content_before or None
```

验证:作者因 Windows 检出缺少 `transformers`(`ModuleNotFoundError: No module named 'transformers'`)未能本地运行目标 pytest,仅执行了 `git diff --check HEAD~1..HEAD`、ruff check/format 与 py_compile。

## 影响与回归风险
- 影响窗口:正文未给出明确版本/提交范围;背景为 MiniMax M3 经 #45381 落地 `main` 之后的流式 reasoning 解析路径。
- 回归面:按 diff,MiniMax M3 解析器的流式提取(`extract_reasoning_streaming`)、reasoning-end 判定(影响工具调用启动时机)、content token 提取与 reasoning token 计数全部重写为序列匹配逻辑;非流式 `extract_reasoning` 在 diff 中未改动;测试侧 `run_streaming` 改用 `encode_runtime`。

## 关联
- Fixes #45687
- 原 PR #45713(base `m3_release` 被删后自动关闭)
- MiniMax M3 落地 main:#45381

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
