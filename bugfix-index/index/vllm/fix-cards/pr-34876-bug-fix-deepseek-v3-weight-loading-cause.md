# PR #34876: [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34876](https://github.com/vllm-project/vllm/pull/34876) |
| 作者 | wzhao18 |
| 合并日期 | 2026-02-19 |
| labels | bug, deepseek, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
DeepSeek V3 权重加载因 QKV-A 投影的 prefix 拼接错误而失败,修正构造参数与加载前缀后恢复。

## 问题现象
正文未描述具体现象,仅在 Purpose 中写明修复 #34869。给出的复现/验证命令为 `vllm serve nvidia/DeepSeek-V3.1-NVFP4 -tp=4`,即 DeepSeek V3.1 NVFP4 权重在 4 卡张量并行下加载时出错(从标题看是 weight loading 失败)。

## 根因
正文未披露根因,正文仅一句话「Fix #34869」。

从 diff 描述改动所暗示的机制(由 diff 推断):`vllm/model_executor/models/deepseek_v2.py` 中 `DeepSeekV2FusedQkvAProj`(融合的 QKV-A 投影层)存在两处问题——其一是 `__init__` 的 `output_size` 参数类型仍是单个 `int`,而融合层实际需要 `list[int]`(各子层输出维度不同);其二是构造 `MergedColumnParallelLinear` 时把加载前缀硬编码为 `f"{prefix}.kv_a_proj_with_mqa"`,即仍然沿用拆分前的单层权重名,导致按 checkpoint 权重名去匹配时前缀不正确、权重加载失败或加载错位。修复后前缀直接使用传入的 `prefix`,即权重名由调用方按融合结构给出。

## 修复方案
改 `vllm/model_executor/models/deepseek_v2.py` 中 `DeepSeekV2FusedQkvAProj.__init__`:
1. `output_size: int` 改为 `output_size: list[int]`,匹配融合投影的多输出维度。
2. 内部构造线性层时的 `prefix=f"{prefix}.kv_a_proj_with_mqa"` 改为 `prefix=prefix`,去掉硬编码后缀,让权重加载名与 checkpoint 对齐。

关键 diff 摘录:
```diff
-        output_size: int,
+        output_size: list[int],
...
-            prefix=f"{prefix}.kv_a_proj_with_mqa",
+            prefix=prefix,
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅改 `deepseek_v2.py` 一个类,影响所有走 `DeepSeekV2FusedQkvAProj` 融合 A 投影路径的 DeepSeek V2/V3 系列模型的权重加载;若外部调用方仍按旧的 `int` 类型传 `output_size`,会在构造时报错。未触及量化内核与推理逻辑。

## 关联
[#34869](https://github.com/vllm-project/vllm/issues/34869)(正文 Purpose 中引用的修复目标)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
