# PR #47953: [Bugfix][Spec Decode] Restrict embedding-width share guard to EAGLE drafts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47953](https://github.com/vllm-project/vllm/pull/47953) |
| 作者 | evantakahashi |
| 合并日期 | 2026-07-21 |
| labels | bug, speculative-decoding, ready, v1, verified |
| 热度 | 👍 5 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
Gemma4 MTP 因 embedding 宽度守卫错误禁用权重共享而在初始化时崩溃;守卫改为仅对 EAGLE draft 生效,MTP 恢复无条件共享。

## 问题现象
Fixes #47794:Gemma4 MTP 在 nightly 上引擎初始化失败,报线性层形状不匹配 `[s47, 3840] x [5632, 1024]`。触发条件是 draft 与 target 的 embedding 宽度不一致的 MTP 模型(如 Gemma4 MTP),在 `_maybe_share_embeddings` 处即崩溃。

## 根因
正文完整披露:#43957 在 `_maybe_share_embeddings` 中加入了 embedding 宽度守卫,宽度不同就禁用 target↔draft embedding 共享。这对 EAGLE draft 是正确的——EAGLE 在自己的 hidden size 上消费 `input_embeds`(即 #43957 修的 `Eagle3MiniMaxM2` XPU 场景);但对 MTP draft 是错的——Gemma4 MTP 的 `pre_projection` 构造为 `Linear(2 * backbone_hidden_size, hidden_size)`,其 draft 宽度的 `embed_tokens` 只用于填充 tied draft 维 `lm_head`,`gemma4_mtp.py` 明确写了共享时它会被 backbone 宽度的表替换。共享被禁用后,concat 宽度变成 `1024 + 2816 = 3840` 而不是 `2816 + 2816 = 5632`,初始化即崩溃。

## 修复方案
改 `vllm/v1/spec_decode/llm_base_proposer.py` 的 `_maybe_share_embeddings`:宽度守卫只在 EAGLE draft 上生效,以该方法中现成的 EAGLE/MTP 判别属性 `has_own_embed_tokens` 为键;MTP draft 恢复 #43957 之前的无条件共享,EAGLE 行为保持不变。

```diff
-            if share_embeddings:
+            if share_embeddings and hasattr(self.model, "has_own_embed_tokens"):
+                # EAGLE drafts consume input embeddings at their own hidden
+                # size, so only share when the widths match. MTP drafts
+                # project target-width embeddings (e.g. Gemma4 MTP's
+                # pre_projection takes 2 * backbone_hidden_size), so the
+                # width check does not apply to them.
                 draft_embed = self.model.model.embed_tokens
```

作者用真实 `Gemma4MTP` 模块做了运行时 A/B 验证(CPU 即可):修复前新测试 `test_mtp_shares_embeddings_when_dims_differ` 失败、forward 报 `mat1 and mat2 shapes cannot be multiplied (4x40 and 64x8)`;修复后 4 个测试全过、共享恢复、tied `lm_head` 保持完整;EAGLE 宽度不同仍不共享、宽度相同共享。作者还提到与 #47833 存在重叠(时间上作者先在 issue 认领),本 PR 额外提供运行时验证与 MTP dims-match 回归用例。

## 影响与回归风险
- 影响窗口:#43957 合入之后至本 PR 之前,embedding 宽度不一致的 MTP draft(如 Gemma4 MTP)无法初始化。
- 回归面:v1 spec decode 的 target/draft embedding 共享路径。MTP 回到无条件共享,若存在依赖「宽度不同不共享」行为的 MTP 场景会受影响;EAGLE 路径行为不变。作者未在真实 26B checkpoint 上做端到端 GPU 验证(正文注明求助他人确认)。

## 关联
- #47794(所修 issue)
- #43957(引入宽度守卫的 PR)
- #47833(同问题竞争修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
