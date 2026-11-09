# PR #20030: Revert "Fix(models/siglip): Add compatibility for Gemma models quantized by llm-compressor"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20030](https://github.com/vllm-project/vllm/pull/20030) |
| 作者 | Isotr0py |
| 合并日期 | 2025-06-25 |
| labels | ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
回滚 #19643:该兼容性修复破坏了未量化 Gemma3 模型的加载,先回滚等正式修复。

## 问题现象
据 PR 正文,上游 #19643(为 llm-compressor 量化的 Gemma 模型在 Siglip 中加兼容)被报告破坏了未量化(unquantized)Gemma3 模型的加载,即安装该改动后,原本可正常加载的未量化 Gemma3 模型会加载失败。正文未给出具体报错日志与触发参数。

## 根因
正文披露:#19643 的改动破坏了未量化 Gemma3 模型的加载("It's reported that #19643 broke unquantized Gemma3 models loading"),作者据此回滚。更细的机制:被回滚的改动在 `gemma3_mm.py` 的权重映射表里新增了 `"vision_tower.vision_model.": "vision_model."` 这一条映射,该映射对未量化模型同样生效,可能把不该改写的权重前缀也改写掉,导致加载失败(由 diff 推断)。正文未披露更完整的根因分析。

## 修复方案
回滚 PR #19643,直接删除 `vllm/model_executor/models/gemma3_mm.py` 中 `hf_to_vllm_mapper` 映射表里新增的那一行:

```diff
             "lm_head.": "language_model.lm_head.",
-            "vision_tower.vision_model.": "vision_model.",
         })
```

即恢复到该兼容性修复合入前的权重映射,未量化 Gemma3 恢复原加载路径;llm-compressor 量化 Gemma 的兼容问题留待正式修复。

## 影响与回归风险
- 影响窗口:#19643 合入到本回滚之间使用了 Gemma3 多模态模型的版本(正文未给出具体版本号)。
- 回归面:改动仅删除一条权重映射,位于 Gemma3 多模态模型的权重加载路径。llm-compressor 量化的 Gemma 模型可能重新回到 #19643 之前无法正常加载的状态(正文未描述该场景的具体表现);未量化 Gemma3 不受本回滚影响。

## 关联
- vllm-project/vllm#19643(被回滚的 PR)
- https://github.com/vllm-project/llm-compressor/issues/1546#issuecomment-3000303239(后续正式修复的进展讨论)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
