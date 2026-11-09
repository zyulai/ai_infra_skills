# PR #25925: [Model][Bugfix] Fix MiDashengLM audio encoder mask by removing incorrect `logical_not`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25925](https://github.com/vllm-project/vllm/pull/25925) |
| 作者 | zhoukezi |
| 合并日期 | 2025-09-30 |
| labels | ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
MiDashengLM 音频编码器注意力从手写实现换成 SDPA 后,残留了一个多余的 mask `logical_not` 取反(SDPA 掩码语义相反,本应删掉),导致 encoder 输出错误 embeddings;删掉取反即修复。

## 问题现象
MiDashengLM 的音频编码器产生错误的 embeddings。正文未描述更具体的线上症状或触发面,问题源自 PR #25854 对 MiDashengLM 音频编码器 attention 的修复在合并前的实现替换。

## 根因
正文披露:#25854 最初用手写 attention 实现修复问题,按 review 意见在合并前换成了基于 SDPA 的实现。SDPA 对 attention mask 的解释与原手写代码正好相反(掩码语义相反),为保持一致本应同时删掉对 mask 的逻辑取反,但这一调整在提交的改动中被遗漏,残留的 `logical_not` 使掩码含义反转,音频编码器因此输出错误 embeddings。

## 修复方案
改 `vllm/model_executor/models/midashenglm.py`:构造 attention mask 后,不再对 mask 做 `logical_not()`,直接 `mask.split(target_length_in_patches, dim=-1)` 得到 split_masks;移除这个遗留的取反操作,恢复 encoder 正确输出。

```diff
             mask = self._to_mask(max_length=t, lengths=scaled_lengths)
-            split_masks = mask.logical_not().split(target_length_in_patches,
-                                                   dim=-1)
+            split_masks = mask.split(target_length_in_patches, dim=-1)
```

## 影响与回归风险
- 影响窗口:自 PR #25854 合入(引入 SDPA 实现与残留取反)之后、本 PR 之前。
- 回归面:仅 `midashenglm.py` 音频编码器的 mask 构造路径,改动面极小(一行)。所有使用 MiDashengLM 音频编码器的模型输出会由错误变为与手写实现语义一致的正确结果;不涉及其他模型或通用 attention 代码。

## 关联
- [PR #25854](https://github.com/vllm-project/vllm/pull/25854)(引入问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
