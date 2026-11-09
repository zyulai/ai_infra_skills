# PR #33892: [Fix] Speculative decoding crashes with DP-Attention

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33892](https://github.com/sgl-project/sglang/pull/33892) |
| 作者 | ekzhang |
| 合并日期 | 2026-08-09 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
EAGLE-3 + DP-Attention 在 idle batch 上对空 seq_lens 取 max 崩溃;idle 批次跳过 init_forward_metadata。

## 问题现象
EAGLE-3 worker 开 DP attention,某 DP rank 处于 idle batch(本轮无请求)时 scheduler 抛异常,正文附栈:`[DP0 TP0 EP0] can't use cuda graph for draft extend! may have correctness issue!`,异常点在 `multi_layer_eagle_worker_v2.py` 的 `_draft_extend_for_decode` → `flashattention_backend.py` `init_forward_metadata` 的 `seq_lens_cpu.max().item()`,报 `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0.`。触发条件:投机解码 + DP attention + decode 阶段的 draft extend 遇到 idle batch。

## 根因
正文 Motivation:修复「speculative decoding and DP attention on EAGLE-3 workers with an idle batch」的崩溃,修改方式为「Conditioned the metadata generation code path on non-idle batch」。(由 diff 推断)DP attention 的 idle 轮次该 rank 没有任何请求,seq_lens 为空;draft extend 仍对其调用 `init_forward_metadata`,内部 `seq_lens_cpu.max().item()` 对 numel==0 的张量抛错;diff 注释亦写明 idle 轮「has nothing to plan pre-pad」。

## 修复方案
`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` 的 `_draft_extend_for_decode`:仅当 `not forward_batch.forward_mode.is_idle()` 时才调用 draft runner 的 `attn_backend.init_forward_metadata(forward_batch)`;idle 批次跳过 metadata 初始化后继续执行 draft forward。

```diff
-                self.draft_runner_list[step].attn_backend.init_forward_metadata(
-                    forward_batch
-                )
+                if not forward_batch.forward_mode.is_idle():
+                    # An idle round (DP attention: this rank has no requests) has nothing
+                    # to plan pre-pad. Avoid raising when seq_lens is empty here.
+                    self.draft_runner_list[step].attn_backend.init_forward_metadata(
+                        forward_batch
+                    )
```

## 影响与回归风险
- 回归面:只改 `_draft_extend_for_decode` 中 metadata 初始化的调用条件;非 idle 批次路径完全不变。idle 批次现在在未初始化 attention metadata 的状态下继续 draft forward,隐含「idle 轮的 draft 输出不被使用」的假设(由 diff 推断);其他投机 worker 实现(非 multi_layer v2)不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
