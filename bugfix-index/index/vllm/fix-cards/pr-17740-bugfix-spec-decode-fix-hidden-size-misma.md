# PR #17740: [BugFix][Spec Decode] Fix hidden size mismatch between target and eagle head

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#17740](https://github.com/vllm-project/vllm/pull/17740) |
| 作者 | WoosukKwon |
| 合并日期 | 2025-05-07 |
| labels | speculative-decoding, ready, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
Eagle 投机解码从目标模型配置取 hidden_size,目标与 eagle head 隐藏维不一致(如 Llama 3.3 70B)时出错;改为从 draft 模型配置获取。

## 问题现象
正文未描述具体现象(无日志或报错),只说明该 bug 在目标模型与 eagle head 的 hidden size 不匹配时出现,典型例子是 Llama 3.3 70B;影响使用 EAGLE 类投机解码且 draft/eagle head 与目标模型隐藏维不同的配置。

## 根因
正文披露根因为目标模型与 eagle head 的 hidden size 不匹配;但未展开机制。改动所暗示的机制(由 diff 推断):`vllm/v1/spec_decode/eagle.py` 的 `__init__` 原先用 `vllm_config.model_config.get_hidden_size()`(目标模型)取 hidden_size,并用它分配 `self.hidden_states` 缓冲;当 draft 模型(eagle head)的 hidden size 与目标模型不同时,缓冲维度与 draft 侧实际 hidden states 维度不一致。diff 中的注释也写明:"the draft model's hidden size can be different from the target model's hidden size (e.g., Llama 3.3 70B)"。

## 修复方案
`vllm/v1/spec_decode/eagle.py` `__init__`:先保存 `self.speculative_config` 与 `self.draft_model_config`,hidden_size 改从 draft 模型配置获取:

```diff
-        self.hidden_size = vllm_config.model_config.get_hidden_size()
+        # We need to get the hidden size from the draft model config because
+        # the draft model's hidden size can be different from the target model's
+        # hidden size (e.g., Llama 3.3 70B).
+        self.hidden_size = self.draft_model_config.get_hidden_size()
```

`self.hidden_states` 缓冲即按 draft 的 hidden size 分配;其余为代码梳理(属性赋值重排、删空行),不改行为。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响 v1 EAGLE 投机解码路径的 hidden_size 来源;对目标模型与 eagle head hidden size 一致的常见配置,取值不变(无行为差异);对不一致的配置,hidden_states 缓冲改按 draft 维度分配。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
