# PR #35368: [Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35368](https://github.com/vllm-project/vllm/pull/35368) |
| 作者 | linyueqian |
| 合并日期 | 2026-02-26 |
| labels | bug, multi-modality, qwen, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
#33605 使 Qwen-Omni 非交错 embed 路径绕过父类逻辑,混模态下音频内容识别失效,恢复 super().embed_input_ids() 修复。

## 问题现象
Qwen2.5-Omni 在混合模态(audio + image + video)输入下不再正确工作:模型无法识别音频内容——e2e 中模型称 unable to recognize the audio,修复后正确识别为 "Mary and her little lamb"。Qwen3-Omni 存在同样模式,一并修复。对应 issue #34506。

## 根因
正文披露:回归由 #33605 引入——它把两个模型 `embed_input_ids` 的非交错路径改为直接调用 `_merge_multimodal_embeddings`,不再经过 `super().embed_input_ids()`。

## 修复方案
`qwen2_5_omni_thinker.py` 与 `qwen3_omni_moe_thinker.py` 的 `embed_input_ids` 非交错默认路径恢复为父类实现:

```diff
-        return _merge_multimodal_embeddings(
-            inputs_embeds, multimodal_embeddings, is_multimodal
+        return super().embed_input_ids(
+            input_ids,
+            multimodal_embeddings=multimodal_embeddings,
+            is_multimodal=is_multimodal,
+            handle_oov_mm_token=handle_oov_mm_token,
         )
```

交错的 `use_audio_in_video` 路径不变,仍走 `merge_interleaved_embeddings`;Qwen3-Omni 的 deepstack 处理(经 `_set_deepstack_input_embeds` 副作用保存)在最终 merge 之前完成,不受影响。新增无需 GPU/权重的单元测试 `tests/model_executor/test_qwen2_5_omni_embed.py`。

## 影响与回归风险
- 影响窗口:正文说明回归自 #33605 起。
- 回归面:两个 Omni 模型的 `embed_input_ids`;非交错路径回到父类实现,`handle_oov_mm_token` 语义随之恢复;交错路径与 deepstack 行为不变。

## 关联
[#34506](https://github.com/vllm-project/vllm/issues/34506)、[#33605](https://github.com/vllm-project/vllm/pull/33605)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
