# PR #19401: Fix nightly VLM accuracy: gemma3n TP fixes + removal, latency thresholds

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19401](https://github.com/sgl-project/sglang/pull/19401) |
| 作者 | alisonshao |
| 合并日期 | 2026-02-27 |
| labels | Multi-modal |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
gemma-3n-E4B-it 因 6 处 TP 并行 bug 无法跑 TP=2;把 AltUp/投影等小层改为 ReplicatedLinear、投影层 gather_output=True 并移除与 JIT RoPE 不兼容的 positions.unsqueeze(0),另放宽 MiMo-VL 延迟阈值。

## 问题现象
nightly VLM 精度测试(2-GPU runner)失败。`google/gemma-3n-E4B-it` 单卡 OOM,需要 `--tp=2` 才能跑,但模型代码存在 6 处 TP bug 导致 TP=2 无法正常工作;另 `XiaomiMiMo/MiMo-VL-7B-RL` 测试观察到延迟 33.9s,超过原阈值 32.0s 导致误报失败。正文链接了示例失败 run(actions/runs/22422538348/job/64923364850)。

## 根因
正文披露的 gemma3n TP bug(6 处,跨文本与视觉模型):小尺寸 AltUp 相关层(`correction_coefs`、`prediction_coefs`、`modality_router`、`per_layer_input_gate`、`per_layer_projection`)被错误地做成并行线性层而非 `ReplicatedLinear`;`per_layer_model_projection`、`altup_projections`、`altup_unembed_projections` 缺少 `gather_output=True`;`embedding_projection`(Gemma3nMultimodalEmbedder)原为 `RowParallelLinear` 却接收来自 VocabParallelEmbedding/vision tower 的全尺寸输入;`positions.unsqueeze(0)` 与 SGLang JIT RoPE 不兼容。MiMo-VL 部分根因正文未披露,只是观察到 33.9s 超过 32.0s 阈值而放宽。

## 修复方案
1. `python/sglang/srt/models/gemma3n_causal.py`:五个小层改 `ReplicatedLinear`(correction_coefs、prediction_coefs、modality_router、per_layer_input_gate、per_layer_projection);`per_layer_model_projection`、`altup_projections`、`altup_unembed_projections` 加 `gather_output=True`;forward 中删除:

```python
-        if positions.dim() == 1:
-            positions = positions.unsqueeze(0)
```

2. `python/sglang/srt/models/gemma3n_mm.py`:`Gemma3nMultimodalEmbedder.embedding_projection` 由 `RowParallelLinear` 改为 `ReplicatedLinear`。
3. `test/registered/eval/test_vlms_mmmu_eval.py`:gemma-3n-E4B-it 条目加 `extra_args=["--tp=2"]`;MiMo-VL-7B-RL 延迟阈值由 32.0s 放宽到 40.0s(观测 33.9s)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 gemma3n 系模型的权重切分方式与 MM 测试配置:改 ReplicatedLinear 的层在 TP 下每卡持有完整权重(显存换正确性),`gather_output=True` 的投影层在输出处多一次 all-gather;单卡(TP=1)路径数值行为不变。MiMo-VL 阈值放宽只是测试判定变化,不影响服务行为。

## 关联
- 示例失败 run:https://github.com/sgl-project/sglang/actions/runs/22422538348/job/64923364850

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
