# PR #31988: Fix reward/classification models broken by `load_weights` v2 dispatch (#28671)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31988](https://github.com/sgl-project/sglang/pull/31988) |
| 作者 | Jialin |
| 合并日期 | 2026-07-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
load_weights v2 重构后 reward/classification 模型加载抛 AttributeError、启动即崩,改为直接调用 `_legacy_load_weights` 恢复旧路径。

## 问题现象
PR #28671 把 `LlamaForCausalLM`/`Qwen2ForCausalLM` 的 `load_weights` 重构为向 `self._legacy_load_weights` / `self._load_weights_v2` 分发之后,所有基于 Llama 或 Qwen2 的 reward/classification 模型在启动加载权重时崩溃。正文举例:加载 `LxzGordon/URM-LLaMa-3.1-8B` 时启动报错 `'LlamaForSequenceClassificationWithNormal_Weights' object has no attribute '_legacy_load_weights'`。该失败发生在默认路径(v2 为 opt-in),影响面是全部这四类 wrapper 构建的模型。

## 根因
正文披露:reward/classification wrapper(`llama_reward`、`llama_classification`、`qwen2_rm`、`qwen2_classification`)只继承 `nn.Module`,通过非绑定调用 `Base.load_weights(self, ...)` 借用基座模型的加载器。#28671 把 `load_weights` 原函数体改名为 `_legacy_load_weights` 并加分发逻辑后,wrapper 的 `self` 上并没有 `_legacy_load_weights` 属性,于是默认路径下权重加载抛 `AttributeError`。

## 修复方案
把四个受影响 wrapper 中对基座加载器的非绑定借用从 `Base.load_weights(self, ...)` 改为 `Base._legacy_load_weights(self, ...)`,并在每个调用点加一行注释说明为何直接调 legacy loader,避免日后被"改回" `load_weights`。涉及文件:`llama_reward.py`、`llama_classification.py`、`qwen2_rm.py`、`qwen2_classification.py`。gemma2 / internlm2 / mixtral 系 wrapper 有意不动——其基座模型未转 v2 分发,`load_weights` 仍自包含可用。

关键 diff(以 llama_reward.py 为例):
```diff
     def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        return LlamaForCausalLM.load_weights(self, weights)
+        return LlamaForCausalLM._legacy_load_weights(self, weights)
```

## 影响与回归风险
- 影响窗口:正文指出回归由 #28671 引入(该 PR 将 load_weights 重构为 v2 分发)至本修复落地之间;未给出具体提交区间。
- 回归面:改动仅限四个 wrapper 文件的权重加载入口,恢复 #28671 之前逐字相同的 legacy 加载路径,正文称无输出变化;影响的是 Llama/Qwen2 系 reward 与 classification 模型的加载路径。gemma2/internlm2/mixtral wrapper 不在改动范围内。现有测试 `test_reward_models.py`、`test_score_engine.py`、`test_pooled_hidden_states.py` 在修复前的 trunk 上失败、修复后通过。

## 关联
- [#28671](https://github.com/sgl-project/sglang/pull/28671)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
