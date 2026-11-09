# PR #45818: [Bugfix]: Fix unquantized gpt-oss weight loading broken by FusedMoE r…

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45818](https://github.com/vllm-project/vllm/pull/45818) |
| 作者 | Priyjain-amd |
| 合并日期 | 2026-06-23 |
| labels | bug, gpt-oss, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
FusedMoE 重构改名 expert 参数致 unquantized gpt-oss 加载 KeyError;改用共享 remap_moe_expert_weights 重映射。

## 问题现象
加载标准(非 mxfp4/非 quark)gpt-oss checkpoint 时,unquantized 权重加载路径抛 KeyError(正文描述)。

## 根因
正文披露:FusedMoE/MoERunner inversion 重构(#41184)把 expert weight 参数移入嵌套的 `RoutedExperts` 子模块,参数名从 `...mlp.experts.<w>` 改为 `...mlp.experts.routed_experts.<w>`;unquantized 加载路径 `_load_weights_other` 仍用原始 checkpoint 名索引 `params_dict`,导致 KeyError。

## 修复方案
`vllm/model_executor/models/gpt_oss.py` `_load_weights_other`:改走共享 helper `remap_moe_expert_weights`(Quark 路径已在用),替代本地字符串替换:
```python
# Use centralized weight remapping for MoE expert parameters.
# The FusedMoE refactor moved expert params under
# `mlp.experts.routed_experts.*`; this remaps checkpoint names so
# MoE weight/bias keys resolve against params_dict.
for name, weight in remap_moe_expert_weights(weights, params_dict):
    # Skip layers on other devices.
    if is_pp_missing_parameter(name, self):
        continue
```
helper 只 remap 已知 expert 参数,且仅当嵌套名存在于 `params_dict` 时才改写,保持与旧 checkpoint 布局的向后兼容。

## 影响与回归风险
- 影响窗口:#41184 重构之后的提交(正文以引用方式给出该重构 PR,未给出具体版本范围)。
- 回归面:gpt-oss unquantized 权重加载路径;Quark 路径共用的 helper 逻辑不变;旧 checkpoint 布局因 helper 的存在性检查保持兼容。

## 关联
issue #45830;重构来源 #41184

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
