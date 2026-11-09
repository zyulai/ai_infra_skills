# PR #45067: [Bugfix]: Fix Quark gpt-oss weight loading broken by FusedMoe refactor

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45067](https://github.com/vllm-project/vllm/pull/45067) |
| 作者 | Rohan138 |
| 合并日期 | 2026-06-11 |
| labels | bug, gpt-oss, ready |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
FusedMoe 重构把 router 权重/偏置挪到 `RoutedExperts` 下,导致 Quark 量化的 gpt-oss 权重加载键名失配;在加载路径上把旧键名重映射到新路径修复。

## 问题现象
正文未描述具体现象。按 PR 标题与 Purpose,问题是 Quark 量化的 gpt-oss 模型权重加载被 FusedMoe 重构破坏(weight loading broken)。正文未给出报错信息或复现步骤。

## 根因
PR 正文披露:该问题由 #41184 引入(broken by #41184);重构后 router 的 weight 和 bias 被移到 `RoutedExperts` 类之下,导致旧 checkpoint 的参数键名与 `params_dict` 中的新路径不再匹配,Quark 权重加载失败。

## 修复方案
在 `vllm/model_executor/models/gpt_oss.py` 的权重加载映射逻辑中(`_get_moe_weight_dtype` 所在分支之后),对 fused_name 做前缀重映射,把旧 checkpoint 键名改写为新结构下的键名,使 `w2_bias` 之类的键能解析到 `params_dict`:

```diff
+                # The MoE refactor (#41184) moved expert params under
+                # `mlp.experts.routed_experts.*`; remap the legacy checkpoint
+                # name so keys like w2_bias resolve against params_dict.
+                fused_name = fused_name.replace(
+                    ".mlp.experts.", ".mlp.experts.routed_experts."
+                )
```

即仅改动一处字符串替换,将 `.mlp.experts.` 映射为 `.mlp.experts.routed_experts.`,不改权重本身。

## 影响与回归风险
- 影响窗口:自 #41184 合入后的版本受影响(正文明确说 broken by #41184)。
- 回归面:改动只在 gpt_oss.py 的 Quark 权重加载路径上做键名替换;凡是命中该分支的 fused_name 都会加 `routed_experts` 前缀,若存在本就使用新键名的 checkpoint 或其他分支依赖原键名,可能受影响。正文 Test Plan 与 Test Result 均为空,未提供验证信息。

## 关联
正文提到 #41184(broken by),未给出链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
