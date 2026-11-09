# PR #42660: [Bugfix] Fix incorrect chat template format for Qwen3.5

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42660](https://github.com/vllm-project/vllm/pull/42660) |
| 作者 | DarkLight1337 |
| 合并日期 | 2026-05-15 |
| labels | bug, qwen, ready |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Qwen3.5 聊天模板被误判为 `string` 格式而非 `openai`:多模态与文本输入间多插一个换行、默认不启用交错;修 AST 检测逻辑并加测试用例。

## 问题现象
Qwen3.5 的聊天模板内容格式被检测为 `string` 而不是 `openai`,带来两个可见影响:(1) 多模态输入与文本输入之间被插入一个多余换行;(2) 默认不应用 interleaving(除非用户显式设置 `--interleave-mm-strings`)。作者是在调试用 [speculators#495](https://github.com/vllm-project/speculators/pull/495) 跑 Qwen3.5 的 token mismatch 问题时发现的,并感慨「之前一直没人注意到」。正文 Test Plan / Test Result 两节为空,未提供评测数据。

## 根因
正文披露:Qwen3.5 的聊天模板内容格式检测结果错误(`string` 而非 `openai`),导致上述换行与不交错两个后果;未进一步解释检测为何出错。
(由 diff 推断)检测实现在 `vllm/renderers/hf.py` 的 `_iter_nodes_assign_content_item`:它遍历模板 Jinja AST,寻找 `{%- for content in message['content'] -%}` 形态的循环(即循环迭代器是对 `message['content']` 取下标的节点)来识别 openai 风格 content 列表;而 Qwen3.5 的模板使用 `{%- for item in content -%}` 形态(迭代器是裸的 `Name` 节点 `content`),不匹配原有规则,于是格式被判为 `string`。修复在该函数中追加:当循环迭代器是名为 `content` 的 `jinja2.nodes.Name` 时也 yield 该循环,并在注释中补充「or {%- for item in content -%} loops」。

## 修复方案
- `vllm/renderers/hf.py::_iter_nodes_assign_content_item`:在原有「`message['content']` 下标迭代」匹配之外,新增对裸 `content` 迭代器的匹配:
  ```python
  if isinstance(loop_iter, jinja2.nodes.Name) and loop_iter.name == "content":
      assert isinstance(loop_target, jinja2.nodes.Name)
      yield loop_ast, loop_target.name
  ```
- `tests/renderers/test_hf.py` 的格式判定参数化用例列表中加入 `("Qwen/Qwen3.5-4B", "openai")`,锁定该模型应判定为 openai 格式。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:所有经 `_iter_nodes_assign_content_item` 做聊天模板格式判定的模型:模板中存在 `for item in content` 形态循环的模型,判定结果可能由 `string` 变为 `openai`(行为变化与 Qwen3.5 相同:多模态/文本拼接不再多插换行、默认启用 interleaving);原有 `message['content']` 形态的匹配规则不变,已正确判定为 openai 的模型不受影响。正文未附 token 级对比或其他模型回归验证。

## 关联
- [speculators#495](https://github.com/vllm-project/speculators/pull/495)(发现该问题的调试场景,正文提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
