# PR #27494: Revert "Fix TRTLLM target verify query metadata (#27473)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27494](https://github.com/sgl-project/sglang/pull/27494) |
| 作者 | merrymercy |
| 合并日期 | 2026-06-07 |
| labels | blackwell |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚在 CI 完成前被强行合并的 #27473,恢复 TRTLLM target verify 分支的 `max_seq_len_q` 赋值,待走完 CI 再重开同内容新 PR。

## 问题现象
正文未描述用户侧 bug 现象;说明回滚原因是 #27473(Fix TRTLLM target verify query metadata)在 CI 完成之前被强行合并(force-merged prematurely before CI completed)。本 PR 的 CI States 也显示 Latest PR Test (Base) 为禁止符号、(Extra) 为失败(x)。

## 根因
正文披露的回滚原因:#27473 未走完 CI 即被强合并,流程不合规;作者计划重新开一个携带相同改动的新 PR 走正规 CI。#27473 本身的功能性根因正文未披露。声明这是一次干净回滚(no new logic)。

## 修复方案
干净回滚,只触及 `python/sglang/srt/layers/attention/trtllm_mha_backend.py` 的 `_apply_cuda_graph_metadata`:在 target verify 分支恢复被 #27473 移除的赋值(由 diff 推断 #27473 曾删除该行):

```diff
             metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
             self._copy_swa_page_table(metadata, page_indices, max_seq_pages)
+            metadata.max_seq_len_q = self.speculative_num_draft_tokens
         elif forward_mode.is_draft_extend(include_v2=True):
```

即 CUDA graph 元数据应用时,target verify 场景的 `max_seq_len_q` 重新取 `speculative_num_draft_tokens`。

## 影响与回归风险
- 影响窗口:正文未说明(#27473 强合并至本回滚之间)。
- 回归面:TRTLLM MHA 后端 CUDA graph 路径下 target verify 的 query metadata(`max_seq_len_q`);diff 只触及该文件这一处;labels 含 blackwell,原 PR 面向的验证环境为 Blackwell 硬件(分类标签,正文未展开)。#27473 想修的 target verify metadata 问题是否回归,正文未评价。

## 关联
- [#27473](https://github.com/sgl-project/sglang/pull/27473)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
