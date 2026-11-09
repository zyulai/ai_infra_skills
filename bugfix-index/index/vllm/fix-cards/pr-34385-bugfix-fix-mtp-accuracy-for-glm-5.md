# PR #34385: [Bugfix] Fix MTP accuracy for GLM-5

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34385](https://github.com/vllm-project/vllm/pull/34385) |
| 作者 | mgoin |
| 合并日期 | 2026-02-12 |
| labels | bug, ready, speculative-decoding, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
GLM-5 等 checkpoint 未在 MTP 层内存 `shared_head.head` 权重,导致 MTP 产生 NaN logits、接受率 0%;显式把目标模型 lm_head 共享给每个 MTP 层修复。

## 问题现象
对 GLM-5 这类 checkpoint(MTP 层内没有像 DeepSeek V3.2 那样冗余存放 `shared_head.head` 权重)开 MTP 投机解码时,产生 NaN logits,draft 完全不被接受。正文实测(4xB300,`zai-org/GLM-5-FP8`,MTP `num_speculative_tokens 1`,gsm8k):
- Before:`Mean acceptance length: 1.00`,`Accepted: 0 tokens`,`Per-position acceptance rate: 0.000`,`Avg Draft acceptance rate: 0.0%`
- After:`Mean acceptance length: 1.95`,`Accepted: 5578 / Drafted: 5841`,`Per-position acceptance rate: 0.955`,`Avg Draft acceptance rate: 95.5%`

## 根因
正文披露:既有 `_maybe_share_lm_head` 只设置 `self.model.lm_head`,但 MTP 的 `compute_logits` 使用每个 MTP 层内部的 `shared_head.head`(ParallelLMHead)。当 checkpoint 没有在该路径存放 lm_head 权重副本时,`shared_head.head` 保持未初始化,产生 NaN logits。DeepSeek-V3.2 因 checkpoint 中冗余复制了该权重而"免费"绕过此问题。

## 修复方案
改 `vllm/v1/spec_decode/eagle.py` 的 `_maybe_share_lm_head()`:在原有 `self.model.lm_head` 共享之后,遍历 `self.model.model.layers`,对每个带 `shared_head.head` 的 MTP 层删除其未初始化的 head,并共享目标模型的 lm_head:
```python
for layer in items:
    sh = getattr(layer, "shared_head", None)
    if sh is not None and hasattr(sh, "head"):
        del sh.head
        sh.head = target_language_model.lm_head
        logger.info("Shared target model lm_head with MTP shared_head.head.")
```

## 影响与回归风险
- 回归面:MTP/EAGLE 投机解码的 lm_head 权重共享逻辑;所有 MTP 层的 `shared_head.head` 统一指向目标模型 lm_head,不再依赖 checkpoint 是否冗余存权重。对 DeepSeek-V3.2 这类 checkpoint 自带权重的模型,行为等价(共享同一份权重);无 `shared_head` 的模型路径不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
