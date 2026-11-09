# PR #8482: Fix incorrect KV cache allocation for MTP models.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8482](https://github.com/sgl-project/sglang/pull/8482) |
| 作者 | lifuhuang |
| 合并日期 | 2025-07-29 |
| labels | (none) |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
MTP 模型的 draft worker 按 target 全部层数分配 KV cache 而实际只需 MTP 层,导致内存误判与 OOM;改用 `num_nextn_predict_layers` 计算。

## 问题现象
作者在为 GLM4.5 调 `mem-fraction-static` 时遇到各种意外行为:1) 开启 MTP 时,所有 GLM4.5 模型不显式指定 `mem-fraction-static` 都无法启动;2) 300B 全量模型在 0.8 时能成功启动,而更轻量的 100B Air 版反而 OOM,必须把 `mem-fraction-static` 调到 0.5 以下。

## 根因
正文披露:为 draft model 创建 KV cache 时,当前按 **所有模型层** 分配,而实际上只需要最后一个 MTP layer。即 draft worker 的 `layer_num` 取的是 target 模型的 `num_hidden_layers` 而不是 draft 实际使用的 MTP 层数,造成 KV cache 预算严重虚高,`mem-fraction-static` 的默认值推算因此完全失真。

## 修复方案
1. `python/sglang/srt/configs/model_config.py`:`__init__` 中新增 `self.num_nextn_predict_layers = getattr(self.hf_text_config, "num_nextn_predict_layers", None)`;
2. `python/sglang/srt/model_executor/model_runner.py` 的 `initialize`:draft worker 且模型有 MTP 层时,`model_num_layers` 取 `num_nextn_predict_layers`,否则取 `num_hidden_layers`;`end_layer` 默认改用 `model_num_layers`,并新增断言「PP is not compatible with MTP models」:

```python
model_num_layers = (
    self.model_config.num_nextn_predict_layers
    if self.is_draft_worker and model_has_mtp_layers
    else self.model_config.num_hidden_layers
)
self.end_layer = getattr(self.model, "end_layer", model_num_layers)
```

3. `init_memory_pool` 两处(MLA 与普通 KV cache)的 `layer_num` 从「非 draft 用 `num_hidden_layers`、draft 用 `hf_config.num_nextn_predict_layers`」的条件表达式统一改为 `self.num_effective_layers`。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:MTP 模型(DeepSeek-V3、GLM4.5 等)draft worker 的 KV cache 分配显著减小,`mem-fraction-static` 默认值推算随之变化;非 MTP 模型 `num_nextn_predict_layers` 为 None,行为不变;新增 PP 与 MTP 不兼容的启动期 assert,PP + MTP 组合从静默错误变为显式报错。修复后 4 个 GLM 4.5 模型无需 `--mem-fraction-static` 覆盖即可按上述命令启动,GSM 分数相当。

## 关联
- https://github.com/sgl-project/sglang/pull/8472

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
