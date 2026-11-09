# PR #47031: [Bugfix] Fix GraniteMoeShared weight loading broken by #41184

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47031](https://github.com/vllm-project/vllm/pull/47031) |
| 作者 | mganczarenko |
| 合并日期 | 2026-07-01 |
| labels | bug, ready, verified |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
#41184 重构后父类期望 experts.{e}.w{n}.weight 键,GraniteMoeShared 仍多插一段 routed_experts 致 KeyError;删掉多余段。

## 问题现象
加载 `GraniteMoeSharedForCausalLM` 权重时抛 `KeyError`,引擎初始化失败。复现环境与日志(`ibm-research/moe-7b-1b-active-shared-experts`,TP=2,Intel B60 XPU):

```
(Worker_TP1) ERROR KeyError: 'layers.0.block_sparse_moe.experts.routed_experts.routed_experts.w13_weight'
(Worker_TP0) ERROR KeyError: 'layers.0.block_sparse_moe.experts.routed_experts.routed_experts.w13_weight'
RuntimeError: Engine core initialization failed.
```

修复后同一测试跑通(KV cache 14.1 GiB,正常生成,SUCCESS)。

## 根因
正文披露:FusedMoE/MoERunner 反转重构(#41184)之后,父类 `_load_weights()` 期望专家权重键形如 `experts.{e}.w{n}.weight`;而 `GraniteMoeShared.load_weights()` 在委托给父类之前仍会在键中插入 `routed_experts`,于是产生重复路径 `experts.routed_experts.routed_experts.{e}.w{n}.weight`,查表时 `KeyError`。正文另说明:#45054 已在 `aria.py`、`qwen3_vl_moe.py`、`step3_text.py`、`step3p5.py` 修过同类问题,但为控制范围明确排除了 `granitemoeshared.py`,本 PR 补上遗漏。

## 修复方案
修改 `vllm/model_executor/models/granitemoeshared.py` 的 `load_weights`:从三处 `.replace()` 替换串中删掉多余的 `routed_experts.` 段,使父类收到其期望的 `experts.{e}.w{n}.weight`:

```diff
-                        f".block_sparse_moe.experts.routed_experts.{e}.w1.weight",
+                        f".block_sparse_moe.experts.{e}.w1.weight",
```

w3、w2 两处同理。

## 影响与回归风险
- 影响窗口:正文未给出具体版本号;回归自 #41184 重构起,至本修复合并前。
- 回归面:仅 `granitemoeshared.py` 的权重名映射;只影响 GraniteMoeShared 模型加载路径,其余沿用 #45054 修复的模型不受影响。

## 关联
- #41184(正文指明的回归引入重构)
- #45054(同类问题的前置修复,当时明确排除 granitemoeshared.py)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
