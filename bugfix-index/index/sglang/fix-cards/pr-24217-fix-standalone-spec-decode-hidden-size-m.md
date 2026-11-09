# PR #24217: fix: STANDALONE spec-decode hidden-size mismatch crash

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24217](https://github.com/sgl-project/sglang/pull/24217) |
| 作者 | brian030128 |
| 合并日期 | 2026-05-11 |
| labels | documentation, high priority, run-ci |
| 热度 | 👍 1 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

STANDALONE 投机解码 extend 相位的 hidden_states 占位误用 draft 形状,改按 target 配置定尺寸,消除 4096/2048 形状崩溃隐患。

## 问题现象

报告者(#24215)用 Llama-3.1-8B(target,hidden=4096)+ Llama-3.2-1B(draft,hidden=2048)跑 STANDALONE 投机解码,在 `EagleDraftInput.merge_batch.torch.cat` 处崩溃:

```
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 4096 but got size 2048
```

正文同时指出:该 0.5.9 崩溃在当前 main 上已不可达,本次修的是残留的接线隐患。

## 根因

正文给出完整诊断:`spec_info.hidden_states` 有两种规范形状,按相位切换——decode 相位是 draft 形状(draft 模型输出,经 `capture_for_decode` 与 draft loop 重写);extend 相位是 target 形状(由 target verify 输出、`EagleDraftExtendInput` 携带)。`capture_for_decode`(2025-03-05 加入)已在 `forward_draft_extend` 结束时把 batch 归一回 draft 形状,所以调度器始终看到 draft 形状。本次修的两处 extend 相位接线此前误用 draft 形状,属潜在 bug:其一,`forward_draft_extend_after_decode` 的 idle fallback 需 `enable_dp_attention=True` 才可达;其二,extend cuda graph runner 有 shape-guard 在形状不匹配时静默跳过 `copy_`,故未显性崩溃。

## 修复方案

两个 extend 相位接线点改为对齐 target 形状(extend 相位的产出方):

1. `python/sglang/srt/speculative/eagle_worker.py`(`forward_draft_extend_after_decode` 的 idle fallback)——占位 `EagleDraftExtendInput` 的 hidden_size/dtype 改从 `target_worker.model_runner.model_config` 取:

```diff
+            target_cfg = self.target_worker.model_runner.model_config
             hidden_size = (
-                self.model_config.hidden_size * 3
+                target_cfg.hidden_size * 3
                 if self.speculative_algorithm.is_eagle3()
                 and self.eagle_use_aux_hidden_state
-                else self.model_config.spec_hidden_size
+                else target_cfg.spec_hidden_size
             )
```

2. `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`——extend cuda graph 缓冲改按 target config 的 `spec_hidden_size` 与 dtype 定尺寸。

decode 相位接线保持 draft 形状不变;正文明确:若把它也翻成 target,反而会让调度器拿 target 形状的 idle batch 与 draft 形状的 running batch 做 cat,精确复现该崩溃。

## 影响与回归风险

- 回归面:仅触及 STANDALONE/EAGLE extend 相位的 idle fallback 与 extend cuda graph 缓冲分配两条路径;decode 相位不动。正文注明按 #21058 任务 1(PR #21434),STANDALONE spec v1 路径将不再携带 `hidden_states`,届时这两处成为死代码。

## 关联

- Fixes #24215
- #21058(任务 1)/ PR #21434

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
