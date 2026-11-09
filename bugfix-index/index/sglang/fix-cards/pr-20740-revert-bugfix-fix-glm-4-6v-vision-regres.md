# PR #20740: Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20740](https://github.com/sgl-project/sglang/pull/20740) |
| 作者 | mickqian |
| 合并日期 | 2026-03-18 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
按维护者要求整体回滚 #20463,移除 glm4v_moe/glm_ocr 权重加载末尾的 copy_conv3d_weight_to_linear 调用。

## 问题现象
正文未描述具体现象。该 PR 是对 sgl-project/sglang#20463("[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr")的整体回滚,动机仅一句:"per request by @JustinTong0323"(应维护者要求回滚)。为何原修复需要撤回,正文未说明。

## 根因
正文未披露根因。

(由 diff 推断)本 PR 撤销 #20463 在 `glm4v_moe.py` 与 `glm_ocr.py` 两处 `load_weights` 末尾新增的调用:

```python
if not is_nextn:
    self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

即该"修复"在非 nextn 分支对 visual.patch_embed 做的 conv3d 权重拷贝被整体移除,模型权重加载行为恢复到 #20463 之前;回滚的具体原因正文未给出。

## 修复方案
纯回滚,共两处对称删除:

- `python/sglang/srt/models/glm4v_moe.py`:`Glm4vMoeForConditionalGeneration.load_weights` 末尾删除 `copy_conv3d_weight_to_linear()` 调用块;
- `python/sglang/srt/models/glm_ocr.py`:`GlmOcrForConditionalGeneration.load_weights` 末尾删除同样的调用块。

```diff
-        if not is_nextn:
-            self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

## 影响与回归风险
- 影响窗口:自 #20463 合并起至本回滚。
- 回归面:GLM-4.6V(GLM-OCR)系两个模型的权重加载路径——`glm4v_moe` 与 `glm_ocr` 的 `load_weights`(非 nextn 分支)不再执行 conv3d 权重拷贝,行为回到 #20463 之前;#20463 原本针对的 GLM-4.6V vision regression 可能随之复现,正文未说明替代修复计划。

## 关联
- 回滚对象:sgl-project/sglang#20463
- 回滚发起:per request by @JustinTong0323

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
