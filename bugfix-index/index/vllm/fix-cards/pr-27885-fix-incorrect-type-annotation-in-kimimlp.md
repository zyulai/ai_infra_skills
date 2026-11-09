# PR #27885: fix incorrect type annotation in KimiMLP

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27885](https://github.com/vllm-project/vllm/pull/27885) |
| 作者 | skyloevil |
| 合并日期 | 2025-10-31 |
| labels | ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
KimiMLP.__init__ 的 quant_config 参数类型注解误写为 QKVParallelLinear,改为 QuantizationConfig,纯类型修复。

## 问题现象
正文未描述具体现象。按正文 Summary 与 Changes,问题为 `KimiMLP.__init__()` 中 `quant_config` 参数的类型注解错误:被写成了 `QKVParallelLinear | None`(一个线性层类),而该参数实际接收的是量化配置对象。此类错误不会触发运行时故障,但会误导类型检查与调用方。

## 根因
正文在 Summary/Changes 中明确说明:`quant_config` 的类型注解被错误写成 `QKVParallelLinear | None`,正确类型应为 `QuantizationConfig | None`;同时文件里为该错误注解引入的 `QKVParallelLinear` import 实际并未被使用。

## 修复方案
改动 `vllm/model_executor/models/kimi_linear.py` 两处:
1. `KimiMLP.__init__` 中 `quant_config` 的注解由 `QKVParallelLinear | None = None` 改为 `QuantizationConfig | None = None`。关键 diff 摘录:

```diff
-        quant_config: QKVParallelLinear | None = None,
+        quant_config: QuantizationConfig | None = None,
```

2. 从 `vllm.model_executor.layers.linear` 的 import 中移除未再使用的 `QKVParallelLinear`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:正文 Impact 明确声明这是 type-only change,不影响任何运行时行为;diff 也仅触及类型注解与一行未使用 import,回归风险极低。KimiMLP(kimi_linear.py)的构造签名与传参逻辑均未改变。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
