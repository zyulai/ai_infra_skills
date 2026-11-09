# PR #9976: [BugFix] Fix incorrect hidden_states_tensor in pd disaggregation + eagle

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9976](https://github.com/sgl-project/sglang/pull/9976) |
| 作者 | ZeldaHuang |
| 合并日期 | 2025-09-17 |
| labels | high priority, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
PD 分离部署 + EAGLE 时 hidden states 应取自 spec_info,当前却取了 target 模型 logits_output 中的错误位置;eagle3 分支改从 spec_info.hidden_states 逐请求取。

## 问题现象
正文未描述具体对外现象(如精度错误表现),仅指出:当同时使用 PD disaggregation 与 eagle 时,本应转移 `spec_info` 中的 hidden states,但当前代码转移的是 target 模型 `logits_output` 中的 hidden states,导致 `hidden_states_tensor` 不正确。

## 根因
正文披露:PD disaggregation + eagle 场景下 hidden states 的来源写错——应该取 `spec_info` 里的,却取了 target model 的 `logits_output`。

## 修复方案
`python/sglang/srt/disaggregation/prefill.py` 的 `process_batch_result_disagg_prefill`:当 `self.spec_algorithm.is_eagle3()` 时,改用 `batch.spec_info.hidden_states[i].cpu().clone()` 作为 `req.hidden_states_tensor`;否则保持原逻辑 `logits_output.hidden_states[last_hidden_index].cpu().clone()`。

关键 diff:
```diff
-                    req.hidden_states_tensor = (
-                        logits_output.hidden_states[last_hidden_index].cpu().clone()
-                    )
+                    if self.spec_algorithm.is_eagle3():
+                        req.hidden_states_tensor = (
+                            batch.spec_info.hidden_states[i].cpu().clone()
+                        )
+                    else:
+                        req.hidden_states_tensor = (
+                            logits_output.hidden_states[last_hidden_index].cpu().clone()
+                        )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 PD prefill 侧的结果处理路径。eagle3 时 hidden states 来源与索引方式改变(eagle3 分支用请求序号 i 直接索引 spec_info.hidden_states,而非 last_hidden_index,由 diff 推断需 spec_info.hidden_states 按请求排布);非 eagle3 路径行为不变。

## 关联
正文未列出其他 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
