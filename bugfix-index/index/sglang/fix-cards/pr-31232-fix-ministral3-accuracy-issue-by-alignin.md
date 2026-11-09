# PR #31232: Fix Ministral3 accuracy issue by aligning YaRN RoPE scaling with Transformers implementation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31232](https://github.com/sgl-project/sglang/pull/31232) |
| 作者 | e-martirosian |
| 合并日期 | 2026-07-15 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
SGLang 的 YaRN RoPE scaling 未处理 `rope_scaling` 中的 `mscale`/`mscale_all_dim`,与 Transformers 行为不一致导致 Ministral3 精度偏低(GSM8K 0.860→0.920);补齐参数并对齐 HF 缩放公式。

## 问题现象
基于 YaRN 的模型(如 Ministral3)在 SGLang 下生成精度明显低于 Transformers 实现:模型 `mistralai/Ministral-3-14B-Instruct-2512-BF16`,TP4,修复前 SGLang 原生实现 GSM8K 0.860、MMMU 0.483;Transformers 实现(`--model-impl transformers`)GSM8K 0.925(MMMU 该配置下 Failed);修复后 SGLang GSM8K 0.920、MMMU 0.523。正文未附具体错误日志。

## 根因
正文披露:SGLang 现有 YaRN RoPE scaling 实现与 Transformers 实现不完全一致,尤其没有处理 `rope_scaling` 中的 `mscale` 与 `mscale_all_dim` 参数,导致 RoPE 缩放行为不同,进而造成 YaRN 类模型(如 Ministral3)精度下降。

## 修复方案
改两个文件:

1. `python/sglang/srt/layers/rotary_embedding/factory.py` `get_rope`:传给 `YaRNScalingRotaryEmbedding` 的 `extra_kwargs` 白名单在原有 `extrapolation_factor/attn_factor/beta_fast/beta_slow` 基础上增加 `mscale`、`mscale_all_dim`,使 `rope_scaling` 配置中的这两个值能传入构造函数。
2. `python/sglang/srt/layers/rotary_embedding/yarn.py` `YaRNScalingRotaryEmbedding.__init__`:新增 `mscale: float = None`、`mscale_all_dim: float = None` 参数;两者都提供时按 Transformers 兼容逻辑重算 `mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)`(引用 HF modeling_rope_utils.py 的 YaRN scaling logic),否则退回原 `yarn_get_mscale_simple(scaling_factor)`;最后统一乘 `attn_factor`。

关键 diff 摘录:

```diff
+        if mscale is not None and mscale_all_dim is not None:
+            # Match Hugging Face's YaRN RoPE scaling (supports mscale/mscale_all_dim)
+            self.mscale = float(
+                yarn_get_mscale(self.scaling_factor, mscale)
+                / yarn_get_mscale(self.scaling_factor, mscale_all_dim)
+            )
+        else:
+            # Get n-d magnitude scaling corrected for interpolation
+            self.mscale = float(yarn_get_mscale_simple(self.scaling_factor))
+        self.mscale *= attn_factor
```

作者验证结果见问题现象一节的表格(GSM8K 0.860→0.920,MMMU 0.483→0.523)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响所有使用 `YaRNScalingRotaryEmbedding` 的模型:配置了 `mscale`+`mscale_all_dim` 的模型(如 Ministral3)mscale 计算公式改变,输出数值变化(向 HF 对齐);未配置这两个参数的模型走 `yarn_get_mscale_simple` 分支,数值与原先一致(原先 `simple*attn_factor` 内联,现拆开相乘,结果相同)。`get_rope` 的 extra_kwargs 白名单扩大,其他 RoPE 类型不受影响。

## 关联
- https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/modeling_rope_utils.py#L412(正文引用的 HF YaRN scaling logic)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
