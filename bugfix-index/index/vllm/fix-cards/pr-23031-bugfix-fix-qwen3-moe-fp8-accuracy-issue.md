# PR #23031: [Bugfix] fix qwen3 moe fp8 accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23031](https://github.com/vllm-project/vllm/pull/23031) |
| 作者 | jinzhen-lin |
| 合并日期 | 2025-08-17 |
| labels | bug, qwen, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
Qwen3 MoE 的 gate 层被误以 fp8 量化(原权重 bf16)导致精度问题,修复为 fp8 配置解析时兼容 `modules_to_not_convert` 忽略列表键名。

## 问题现象
Qwen3 MoE 使用 fp8 量化时出现精度问题(标题所述;正文未展开具体现象与数值)。问题由 issue #22881 报告。

## 根因
正文披露:原始问题由 #22017 引入——`gate` 层被以 fp8 量化初始化,但其原始权重是 bf16(即本应保留高精度的层被错误量化,造成精度损失)。从 diff 描述改动所暗示的机制:Qwen3 MoE 的 checkpoint 配置用 `modules_to_not_convert` 键声明不量化层名单,而 vLLM 的 `Fp8Config.from_config` 此前只读取 `ignored_layers` 键,导致忽略名单取不到、gate 层未被豁免(由 diff 推断)。

## 修复方案
`vllm/model_executor/layers/quantization/fp8.py` 的 `Fp8Config.from_config`:在 `ignored_layers` 为空时回退读取 `modules_to_not_convert` 键:

```python
if not ignored_layers:
    ignored_layers = cls.get_from_keys_or(config,
                                          ["modules_to_not_convert"],
                                          None)
```

其余构造参数(`is_checkpoint_fp8_serialized`、`activation_scheme`、`weight_block_size` 等)不变。

## 影响与回归风险
- 影响窗口:#22017 合入之后、本 PR 之前的 Qwen3 MoE fp8 部署。
- 回归面:所有 fp8 量化配置解析路径——checkpoint 配置里含 `modules_to_not_convert` 键且此前未设 `ignored_layers` 的模型,行为从「全部量化」变为「按名单跳过」;对没有该键的 checkpoint 无影响。已声明 `ignored_layers` 的配置优先级不变(仅在为空时回退)。

## 关联
Fixes #22881;问题由 #22017 引入

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
