# PR #30119: [BugFix] Fix DeepSeek-R1 hang with DP and MTP

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30119](https://github.com/vllm-project/vllm/pull/30119) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-12-09 |
| labels | ready, v1, deepseek |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
DP+MTP 下 dummy_run 误关 drafter 的 cudagraphs、跳过 pad_for_cudagraph 导致挂起,修正判断条件。

## 问题现象
DeepSeek-R1 在同时开启 DP 与 MTP(EAGLE 投机解码)时出现 hang。触发条件为 PR 正文描述的一个边界情况:当 dummy_run 以 FULL cudagraphs 模式运行时,会错误地为 drafter 禁用 cudagraphs,导致 dummy_run 在 `_pad_batch_across_dp` 之后不再调用 `pad_for_cudagraph`,从而挂起。正文未给出具体日志。

## 根因
正文披露:存在一个边界情况(dummy_run 跑 FULL cudagraphs 时),drafter 的 cudagraphs 被错误禁用;由于不在 cudagraphs 路径上,`_dummy_run` 在 `_pad_batch_across_dp` 之后不再执行 `pad_for_cudagraph`,造成挂起。即 drafter 是否启用 cudagraphs 的判断条件在 FULL cudagraphs 场景下判断错误。长期方案应参照 vllm-project/vllm#28579 重构 drafter:先为 cudagraphs pad、再为 DP pad;本 PR 只是临时 hack(作者自己标注 "this is a hack, need to clean up")。

## 修复方案
修改 `vllm/v1/worker/gpu_model_runner.py` 中 `_dummy_run` 内 EAGLE 分支的 `use_cudagraphs` 判定。原逻辑只看 `cudagraph_runtime_mode.has_mode(CUDAGraphMode.PIECEWISE)`,改为区分是否处于 graph capturing 状态:

```diff
                 use_cudagraphs = (
-                    cudagraph_runtime_mode.has_mode(CUDAGraphMode.PIECEWISE)
-                    and not self.speculative_config.enforce_eager
-                )
+                    (
+                        is_graph_capturing
+                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
+                    )
+                    or (
+                        not is_graph_capturing
+                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
+                    )
+                ) and not self.speculative_config.enforce_eager
```

这样 dummy_run 非捕获场景下 FULL cudagraphs 模式也会为 drafter 启用 cudagraphs,保证 `pad_for_cudagraph` 被调用。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动位于 `_dummy_run` 的投机解码(EAGLE)路径,仅改变 drafter 侧 `use_cudagraphs` 的判定,影响所有启用 EAGLE/MTP 投机解码且与 cudagraphs/DP 组合的场景。作者明确说明这是临时 hack,后续需按 #28579 的思路重构 drafter。

## 关联
- vllm-project/vllm#28579(作者建议的长期重构方向)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
