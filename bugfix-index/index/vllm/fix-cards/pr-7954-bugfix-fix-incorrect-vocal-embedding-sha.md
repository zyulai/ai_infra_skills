# PR #7954: [Bugfix] Fix incorrect vocal embedding shards for GGUF model in tensor parallelism

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7954](https://github.com/vllm-project/vllm/pull/7954) |
| 作者 | Isotr0py |
| 合并日期 | 2024-08-29 |
| labels | ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Parallelism / Communication |
| bug 类型 | correctness |

## 一句话结论

GGUF 模型词嵌入在 tp_size>1 时未做权重分片,UninitializedParameter 按完整加载形状物化,导致各 rank 分片错误、logits 计算不对;materialize 前按 output_dim 除以 tp_size 修复。

## 问题现象

正文未描述具体运行现象;从标题与正文可确定:GGUF 模型在 tensor parallelism(`tp_size > 1`)下词嵌入(vocab embedding)分片不正确。作者最初想修的是「GGUF 模型在张量并行下 logits 计算不正确」(关联 #7880),最终定位为 #7520 遗漏了 GGUF 模型词嵌入权重的分片。

## 根因

正文披露:#7520 在支持 GGUF 时遗漏了词嵌入权重在 `tp_size > 1` 时的权重分片(forgotten weight sharding for GGUF model vocal embedding)。即加载 GGUF 词嵌入权重时参数尚为 `UninitializedParameter`,直接按 loaded_weight 的完整形状物化,没有按 TP rank 切分,导致每个 rank 物化了整份(未切分的)权重形状,与 VocabParallelEmbedding 期望的每 rank 分片形状不符。

## 修复方案

修改 `vllm/model_executor/layers/vocab_parallel_embedding.py` 的 `weight_loader`:对 `UninitializedParameter` 物化时,若该参数存在 `output_dim`,先把该维尺寸除以 `self.tp_size` 再 materialize。关键 diff:

```diff
-            param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)
+            shape = list(loaded_weight.shape)
+            if output_dim is not None:
+                shape[output_dim] = shape[output_dim] // self.tp_size
+            param.materialize(tuple(shape), dtype=loaded_weight.dtype)
```

这样 GGUF 词嵌入在 TP 下物化出的就是本 rank 应持有的分片形状,后续拷贝/加载逻辑保持一致。

## 影响与回归风险

- 影响窗口:#7520 合并之后到本修复之前的版本;正文未给出具体版本号。
- 回归面:改动位于 `VocabParallelEmbedding.weight_loader` 的 `UninitializedParameter` 分支,凡走该分支的权重加载都受影响——任何存在 `output_dim` 的参数现在都会按 TP 切分物化;没有 `output_dim` 的参数(如复制到所有 GPU 的权重)行为不变。非 GGUF 或 tp_size=1 场景下除法无效果,行为不变。

## 关联

- FIX #7880
- #7520(引入该遗漏的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
