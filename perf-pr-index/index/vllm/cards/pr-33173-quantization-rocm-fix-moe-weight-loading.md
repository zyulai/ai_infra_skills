# PR #33173: [Quantization][ROCm] Fix MoE weight loading to be robust (Qwen3_MoE/Qwen3_next as example models)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xuebwang |
| 合并 | 2026-01-30 |
| 分类 | MoE(major) |
| Label | documentation, performance, rocm, ready, v1, qwen, kv-connector |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 极小(仅 layer.py 两处各 +3/-1 行) |
| 链接 | https://github.com/vllm-project/vllm/pull/33173 |

## 一句话结论

修复 Qwen3 MoE/Next 权重加载对 0-dim tensor 调 narrow() 的崩溃,打通 FP8 PTPC 与 WMXFP4-AMXFP4 量化 checkpoint 在该路径的可用性。

## 动机

PR 正文给出完整崩溃栈:加载 Qwen3-Next(经在线量化路径 `online_quantization.py`)时,`qwen3_next.load_weights` → `fused_moe/layer.py` 的 `_load_per_channel_weight_scale` → `_load_w13` 调用 `loaded_weight.narrow(...)`,当 loaded_weight 是 0-dim 标量张量时抛 `RuntimeError: narrow() cannot be applied to a 0-dim tensor`,模型加载直接失败。

## 优化点

修复为两处条件收紧:`_load_w13` 与 `_load_w2` 中把 `if not load_full:` 改为 `if not load_full and loaded_weight.ndim > 0:`——0-dim 标量权重(某些 per-channel scale 形态)不再做 TP 分片 narrow,直接整体加载;非 0-dim 路径行为不变。这是健壮性修复而非 kernel 优化,但它决定 Qwen3_MoE/Qwen3_next 在 ROCm 上能否用 FP8 PTPC 与 WMXFP4-AMXFP4(标题与 rocm label 指向此场景)等量化方案起服。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/layer.py: _load_w13/_load_w2 在 narrow 前增加 loaded_weight.ndim > 0 守卫
```

```python
# Only narrow if the loaded_weight is not a scalar (0-dim tensor)
# and we're not loading the full weight
if not load_full and loaded_weight.ndim > 0:
    loaded_weight = loaded_weight.narrow(
        shard_dim, shard_size * tp_rank, shard_size
    )
```

## 性能声明

PR 正文 Test Result 给出两模型 × 两量化方案的数值表,但未注明指标名称与含义,亦无吞吐/延迟数字:Qwen3-30B-A3B-Thinking-2507 为 FP8 PTPC 0.7552、W_MXFP4 A_MXFP4 0.8197;Qwen3-Next-80B-A3B-Instruct 为 FP8 PTPC 0.8892、W_MXFP4 A_MXFP4 0.9074。

## 验证与回归风险

作者勾选的测试覆盖 Qwen3-30B-A3B-Thinking-2507 与 Qwen3-Next-80B-A3B-Instruct 两个模型、FP8 PTPC 与 WMXFP4-AMXFP4 两种量化方案;素材 diff 中未见新增单测。回归风险很低:守卫只改变原本必然崩溃的 0-dim 分支;所有经 `_load_w13`/`_load_w2` 的 MoE 模型共用此路径,标量 scale 权重不参与 TP narrow 的语义与修复前一致(narrow 本就不可用)。

## 关联

素材未给出关联 PR 编号;同区域为 fused_moe 权重加载与 Qwen3 MoE 在线量化(online_quantization)路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
