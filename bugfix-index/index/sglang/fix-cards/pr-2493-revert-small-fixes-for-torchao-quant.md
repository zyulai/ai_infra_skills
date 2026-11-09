# PR #2493: Revert "Small fixes for torchao quant"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2493](https://github.com/sgl-project/sglang/pull/2493) |
| 作者 | merrymercy |
| 合并日期 | 2024-12-16 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#2476 的 torchao 量化改动只对 torch 原生 llama 生效,整体回退恢复原行为。

## 问题现象
正文未描述具体现象,仅说明被回退的 #2476「only works for torch native llama」,即该 torchao 量化修复在 torch 原生 llama 以外的模型上不工作。

## 根因
正文披露:#2476 引入的 torchao 量化修复只适用于 torch native llama,对其他情况无效,因此回退。正文未给出更细的机制分析。(由 diff 推断)被回退的改动有两处:一是 torchao_utils.py 默认 filter_fn 从 `_is_linear(module) and "proj" in fqn` 改回 `"proj" in fqn` 字符串匹配;二是 model_runner.py 把 `apply_torchao_config_to_model` 的调用从 torch TP 应用之前移回之后。回退后 torchao 量化在 torch TP 之后再施加,避免与 torch TP 逻辑相互干扰。

## 修复方案
直接 revert #2476,涉及两个文件:
- `python/sglang/srt/layers/torchao_utils.py`:删除 `from torchao.quantization.quant_api import _is_linear`,filter_fn 恢复为 `return "proj" in fqn`;
- `python/sglang/srt/model_executor/model_runner.py`:`apply_torchao_config_to_model(...)` 调用移回 torch TP 应用(supports_torch_tp 分支)之后。

```diff
-    from torchao.quantization.quant_api import _is_linear
         def filter_fn(module, fqn):
-            return _is_linear(module) and "proj" in fqn
+            return "proj" in fqn
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:使用 torchao 量化的加载路径(torchao_utils 的 filter 逻辑、model_runner 的初始化顺序);torch TP 与 torchao 量化的组合初始化顺序回到 #2476 之前的状态;#2476 原本想修的问题可能随之重现。

## 关联
- sgl-project/sglang#2476(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
