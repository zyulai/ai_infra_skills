# PR #48563: [Bugfix][Gemma4] Fix ModelOpt mixed-precision MoE config mapping

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48563](https://github.com/vllm-project/vllm/pull/48563) |
| 作者 | wangqia0309 |
| 合并日期 | 2026-07-20 |
| labels | bug, quantization, ready |
| 热度 | 👍 2 · 💬 3 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

Gemma4 checkpoint 的 `quantized_layers` 父模块键未重映射到 `.moe.experts`,MoE 查不到 NVFP4 条目回落未量化;新增共享 expert 父模块 mapper 修复。

## 问题现象

ModelOpt `MIXED_PRECISION` 的 Gemma4 checkpoint(其 `quantized_layers` 把 MoE 父模块记为 `model.language_model.layers.N.experts`)在 vLLM 中查不到 NVFP4 条目、MoE 回落到未量化方法。用 [`wangqia0309/gemma-4-26B-A4B-it_nvfp4_experts_fp8_dense_fp8_attn-kv_fp8`](https://huggingface.co/wangqia0309/gemma-4-26B-A4B-it_nvfp4_experts_fp8_dense_fp8_attn-kv_fp8)(NVFP4 experts + FP8 dense/attention + FP8 KV cache)复现并验证:提交者在 RTX 5090 上确认修复后该 checkpoint 可成功启动(正文未附启动失败日志)。

## 根因

正文披露:vLLM 把运行时 `FusedMoE` 模块注册在 `language_model.model.layers.N.moe.experts`;Gemma4 权重迭代器已对每张量 expert 名做了重映射,但该路径不重写 `quantized_layers` 中的父模块键,配置查找因此 miss NVFP4 条目并回落到未量化的 MoE 方法。

## 修复方案

`vllm/model_executor/models/gemma4.py` 定义共享 `_GEMMA4_EXPERT_PARENT_MAPPER`(正则 `(?<!\.moe)\.experts$` → `.moe.experts`,负向断言保证已归一化的 `.moe.experts` 键不被二次改写),以 `|` 组合进 `Gemma4ForCausalLM` 与 `gemma4_mm.py` 中 `Gemma4ForConditionalGeneration` 的 `hf_to_vllm_mapper`;新增两段 mapper 序列的回归测试,验证 expert 父键只归一化一次且后续 causal-only mapping 仍运行。

```diff
+_GEMMA4_EXPERT_PARENT_MAPPER = WeightsMapper(
+    orig_to_new_regex={
+        re.compile(r"(?<!\.moe)\.experts$"): ".moe.experts",
+    }
+)
```

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:Gemma4 两个模型类的 `hf_to_vllm_mapper` 权重名映射——权重加载时名字重写范围扩大到 `.experts` 父模块键;正则负向断言避免对已归一化键二次改写。正文声明本 PR 不改变 Gemma4 GELU 激活行为。

## 关联

#39406、#41988、#47318(duplicate-work check 中提及);HF checkpoint 链接见「问题现象」

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
