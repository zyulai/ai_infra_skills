# PR #28819: [Bugfix] Fix spec decode memory regression after #28549

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28819](https://github.com/vllm-project/vllm/pull/28819) |
| 作者 | zhewenl |
| 合并日期 | 2025-11-20 |
| labels | speculative-decoding, ready, v1, llama, deepseek |
| 热度 | 👍 3 · 💬 7 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#28549 引入 EAGLE 显存回归:L4 24GB 上 OOM;修复 embed_tokens 跳过加载逻辑,并把权重比较从 torch.equal 放宽为 allclose,恢复共享 embedding。

## 问题现象
EAGLE 投机解码出现显存回归,在 24GB 的 L4 GPU 上触发 OOM。测试日志显示修复前模型加载占用 16.4366 GiB 且打出 "Detected EAGLE model with distinct embed_tokens weights. Keeping separate embedding weights from the target model.";修复后降为 15.4581 GiB,并打出 "Detected EAGLE model with embed_tokens identical to the target model. Sharing target model embedding weights with the draft model."。

## 根因
回归由 #28549 引入,正文指出包含两点:
1. `if get_pp_group().world_size == 1 and "embed_tokens." in name:` 的跳过逻辑阻止了 embed_tokens 权重加载,导致权重未初始化,同时内存中还额外保留一份副本。
2. 浮点权重比较误用 `torch.equal()`(应为 `torch.allclose()`):实测因浮点舍入存在约 3e-8 的差异,导致本应判定为相同的 embedding 被判为不同,无法共享权重。

## 修复方案
- `vllm/model_executor/models/deepseek_eagle.py`、`llama4_eagle.py`、`llama_eagle.py`:在 `load_weights` 中删除 PP 单进程时跳过 `embed_tokens.` 的 continue 逻辑(及 llama4_eagle 中对应的未加载断言豁免),恢复 draft 模型自身权重的正常加载。
- `vllm/v1/spec_decode/eagle.py`:`load_model` 判定是否共享 embedding 时改用带容差的比较:

```python
and torch.allclose(
    target_embed_tokens.weight.cpu(),
    self.model.model.embed_tokens.weight.cpu(),
    rtol=1e-5,
    atol=1e-7,
)
```

## 影响与回归风险
- 影响窗口:引入回归的 #28549 合并之后、本 PR 之前的版本;正文以 #28549 为界说明,未给出具体版本号。
- 回归面:EAGLE 投机解码下 llama/llama4/deepseek draft 模型的 `load_weights` 路径与 embedding 共享判定;对 embed 权重存在约 1e-7 量级差异的 checkpoint,现在会按"相同"共享目标模型 embedding(容差内不再保留独立副本)(由 diff 推断)。

## 关联
- #28549

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
