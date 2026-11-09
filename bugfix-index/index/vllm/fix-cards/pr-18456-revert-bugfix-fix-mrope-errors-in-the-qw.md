# PR #18456: Revert "[Bugfix] Fix MRoPE Errors in the Qwen-VL Model When Processing Pure Text (#18407)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18456](https://github.com/vllm-project/vllm/pull/18456) |
| 作者 | DarkLight1337 |
| 合并日期 | 2025-05-21 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
回滚 #18407 对 Qwen-VL 纯文本 MRoPE 的修复,恢复 `_compute_multi_modal_input` 的提前返回与 mrope 输入断言。

## 问题现象
正文未描述具体现象,正文仅链接到 #18407 下的一条评论(作为回滚依据),未给出新的报错信息或复现描述。

## 根因
正文未披露根因(仅提供 #18407 评论链接作为回滚理由)。
(由 diff 推断) #18407 曾为让 Qwen-VL 纯文本请求也能走通,在 `model_runner.py` 的 `_compute_multi_modal_input` 中去掉了「无多模态输入时提前 return」的逻辑,并移除了 mrope 路径对 `image_grid_thw`/`video_grid_thw`/`audio_feature_lengths` 的断言;本 PR 回滚即恢复这两处,说明 #18407 的做法被判定为引入了回归(mrope 模型在无多模态 kwargs 时不应进入该处理路径)。

## 修复方案
恢复 `vllm/worker/model_runner.py` `_compute_multi_modal_input` 中的两段逻辑:无 `mm_kwargs` 时提前返回;mrope 分支重新断言必须带多模态网格信息:

```diff
+        if not mm_kwargs:
+            return
...
+            assert (
+                image_grid_thw is not None or video_grid_thw is not None
+                or audio_feature_lengths is not None), (
+                    "mrope embedding type requires multi-modal input mapper "
+                    "returns 'image_grid_thw' or 'video_grid_thw' or "
+                    "'audio_feature_lengths'.")
```

## 影响与回归风险
- 影响窗口:正文未说明(根因 PR #18407 合入至本回滚之间)。
- 回归面:所有 mrope 模型(Qwen-VL/Qwen2-VL 系等)的输入准备路径;#18407 想修的「纯文本 MRoPE 错误」问题随回滚重新出现,纯文本请求在 mrope 模型上恢复提前返回/断言行为。

## 关联
- #18407(被回滚的 PR,正文链接到其评论)
- 正文未列出其他 issue/PR

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
