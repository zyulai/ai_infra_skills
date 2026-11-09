# PR #42143: fix(eagle3): read norm_before_fc from eagle_config for NVIDIA checkpoint

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42143](https://github.com/vllm-project/vllm/pull/42143) |
| 作者 | ghost |
| 合并日期 | 2026-05-23 |
| labels | speculative-decoding, ready, llama, nvidia, verified |
| 热度 | 👍 28 · 💬 5 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
NVIDIA Eagle3 checkpoint 的 norm_before_fc 藏在 eagle_config 内未被读取,FC 前 RMSNorm 被跳过致接受率下降;改为优先读 eagle_config。

## 问题现象
NVIDIA(GPT-OSS 系列)的 Eagle3 checkpoint 出现接受率下降(degraded acceptance rate),无崩溃、无报错,属静默性能退化。验证:Qwen3-32B + Qwen3-32B_eagle3,1x H200;修复后 eagle3 从 baseline 51.3 tok/s 提升到 67.1 tok/s(1.31x),4/4 greedy 输出与 baseline 完全一致;使用顶层 `norm_before_fc` 的模型无回归。

## 根因
正文披露:这些 checkpoint 把 `norm_before_fc` 存在嵌套的 `eagle_config` dict 里,而非顶层 config 字段;vLLM 目前只检查顶层属性,于是 FC 前的 RMSNorm 被静默跳过,导致接受率下降。

## 修复方案
`vllm/model_executor/models/llama_eagle3.py`:`eagle_config` 取值改为 `getattr(...) or {}` 兜底;`norm_before_fc` 先读 `eagle_config`,回退顶层 attr(与同块 `use_aux_hidden_state` 的既有模式一致):

```python
eagle_config = getattr(self.config, "eagle_config", None) or {}
if "use_aux_hidden_state" in eagle_config:
    self.use_aux_hidden_state = eagle_config["use_aux_hidden_state"]
else:
    self.use_aux_hidden_state = True
self.norm_before_fc = bool(
    eagle_config.get(
        "norm_before_fc", getattr(self.config, "norm_before_fc", False)
    )
)
```

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:所有经 `llama_eagle3.py` 加载的 Eagle3 draft;使用顶层 `norm_before_fc` 的模型走 `getattr` 回退,行为不变(正文称 no regression);`eagle_config=None` 时由 `{}` 兜底,消除原 `eagle_config is not None` 条件分支。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
