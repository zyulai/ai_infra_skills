# PR #673: Revert "Temporary fix invalid sample results"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#673](https://github.com/sgl-project/sglang/pull/673) |
| 作者 | hnyls2002 |
| 合并日期 | 2024-07-20 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #668:移除采样输出非法 token id 时用 argmax 替换的临时修补,恢复原采样错误处理路径。

## 问题现象
正文未描述具体现象,仅一句话说明回滚 #668(Temporary fix invalid sample results)。没有列出非法 token 出现的场景、报错或影响面。

## 根因
正文未披露根因。从 diff 看,#668 曾在 `sample()` 调用 `top_k_top_p_sampling_from_probs` 之后加入一段临时逻辑(机制描述由 diff 推断):

- 构造 `illegal_mask`:`batch_next_token_ids < 0 或 >= probs.shape[-1]`;
- 命中时 `warnings.warn("Illegal token ids in sampling.")` 并用 `torch.argmax(probs, dim=-1)` 替换越界 token。

原代码中该处本就有 `except RuntimeError` 兜底(采样异常时告警并整体回退 argmax),#668 的临时修补在正常返回但含越界 id 的情况下再做替换;本回滚将这段删除。正文未说明移除原因(如治标掩盖采样 kernel 问题等),仅以回滚处置。

## 修复方案
`python/sglang/srt/managers/controller/infer_batch.py` 的 `sample()` 删除 illegal_mask 检查与 argmax 替换段,保留原有结构:

```diff
             batch_next_token_ids, _ = top_k_top_p_sampling_from_probs(
                 probs, uniform_samples, self.top_ks, self.top_ps
             )
-
-            # FIXME: This is a temporary fix for the illegal token ids in sampling.
-            illegal_mask = (
-                batch_next_token_ids < 0 or batch_next_token_ids >= probs.shape[-1]
-            )
-            if torch.any(illegal_mask):
-                warnings.warn("Illegal token ids in sampling.")
-                batch_next_token_ids = torch.where(
-                    illegal_mask, torch.argmax(probs, dim=-1), batch_next_token_ids
-                )
         except RuntimeError as e:
```

## 影响与回归风险
- 影响窗口:正文未说明(#668 合并至本回滚之间)。
- 回归面:采样输出路径——若采样 kernel 再次产出越界 token id,将不再被替换为 argmax 结果,而是原样下发给下游(由 diff 推断);`except RuntimeError` 的整体兜底不受影响。

## 关联
- [#668](https://github.com/sgl-project/sglang/pull/668)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
