# PR #47318: [BugFix] Fix ModelOpt mixed-precision quantization for sparse `quantized_layers` configs.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47318](https://github.com/vllm-project/vllm/pull/47318) |
| 作者 | danielafrimi |
| 合并日期 | 2026-07-07 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
ModelOpt 混合精度的 parent-prefix 回退会让缺失的兄弟层「继承」无关层的量化;改为只对已知 fused 投影按 shard 名推断。

## 问题现象
ModelOpt 混合精度量化下,配置里未列出的层被错误地应用了量化。触发场景:稀疏(sparse)混合 FP8/NVFP4 的 `quantized_layers` 配置——正文以 Nemotron Super NVFP4 为例,说明这类 config 是稀疏且混合精度的,即只给少数层标注 quant_algo。影响面:所有走 ModelOpt 混合精度 dispatch、且 config 中某父层下只列了部分子层的模型。正文未附具体报错日志,问题描述为「missing sibling layers do not inherit quantization from unrelated siblings」这一错误行为。

## 根因
正文披露:`_resolve_quant_algo` 此前的 parent-prefix 回退逻辑会收集某个父层下所有直接子层的 quant_algo 集合,只要集合里恰好只有一种算法,就把该算法返回给当前前缀。于是当父层下只有一个被量化层(例如 `o_proj`)时,缺失的兄弟层(如 `qkv_proj`)也会被推断为同一种量化——把无关层的量化错误地「继承」给了 config 中根本没配置的层。正文明确指出这对稀疏混合 FP8/NVFP4 配置是不安全的(unsafe),因为 NVFP4/FP8 等量化方式会改变层的实现与权重布局,未配置的层被强加量化会破坏混合精度语义。

## 修复方案
改 `vllm/model_executor/layers/quantization/modelopt.py` 的 `_resolve_quant_algo`:删除对父层下「所有子层」的通用收集回退,改为只对已知 packed 投影做显式 shard 查表:
```python
fused_projection_shards = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
shard_names = fused_projection_shards.get(proj_name)
...
if len(shard_algos) == 1:
    return shard_algos.pop()
if len(shard_algos) > 1:
    raise ValueError(...)
```
即仅当请求的层是 `qkv_proj`(查 `q_proj/k_proj/v_proj`)或 `gate_up_proj`(查 `gate_proj/up_proj`)时,才从 `quantized_layers` 直接查这些 shard 前缀的算法;所有 shard 一致才继承,不一致则抛 ValueError;其余层(如独立的 `down_proj`、未配置的 `qkv_proj`)一律返回 None 走未量化路径。`tests/quantization/test_modelopt.py` 新增两个测试:fused `gate_up_proj` 能从 gate/up 的 NVFP4 配置推断出量化;`o_proj→qkv_proj`、`gate_proj→down_proj` 的缺失兄弟层不再被推断(得到 `UnquantizedLinearMethod`)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:仅 ModelOpt 混合精度的算法解析路径:此前「父层下单一量化子层 → 全体子层继承」的行为被移除,依赖该行为、config 中只列 shard 名(而非 vLLM packed 名)的合法 fused 投影场景仍被保留(qkv_proj/gate_up_proj 两条映射);若某 config 曾依赖旧回退把量化强加给未配置层,升级后会变为未量化层,层数值行为与显存占用随之改变;同一 fused 层内 shard 算法不一致时从「静默取一种」变为抛 ValueError。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
