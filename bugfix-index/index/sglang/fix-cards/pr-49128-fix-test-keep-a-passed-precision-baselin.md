# PR #49128: [Fix] test: Keep a passed precision baseline row pointing at its own tensors

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#49128](https://github.com/sgl-project/sglang/pull/49128) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-11-08 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

夜间精度测试失败后重跑通过,baseline store 复用了失败 run 留下的偏差张量作为 baseline;修复为 passed push 必须上传自己的张量。

## 问题现象

`test/registered/debug_utils/test_nightly_precision_regression.py` 在 11 月 7 日 nightly(attempt 1)失败:`zai-org/GLM-5.2-FP8: FAILED - tensor=...model.layers.56.self_attn.inputs.hidden_states rel_diff=0.00117`,同 commit 重跑通过。两层问题:

- 失败本身归因于一台特定 `8-gpu-h200` runner:健康硬件上 dump 张量是 bit 可复现的(同一 commit 在另一台 8x H200 上 16 次全新启动彼此及与 baseline 均 bit 一致,通过的 CI 报 `max_rel_diff=0.0`);该 runner 的偏差(从 layer 48 开始,prefill tokens 11-16,TP rank 1 最大,到 layer 77 增长到 ~1.4e-2)在一个月内跨不同 commit bit 一致,且它与来自其他 runner 的 baseline 的每次比较都失败——没有任何阈值能区分它与真实数值变化。
- 重跑污染了 baseline store:attempt 1 以 `pass_label="failed"` 推送了张量;attempt 2 通过后只刷新 meta 并追加 `passed` manifest 行,该行指向的是 attempt 1 的偏差张量(`fetch` 取最新非 failed 行),下一个健康 runner 上的 nightly 将与之比较。10 月同样的复用已把一次早期偏差变成 baseline。

## 根因

正文披露:`push_run` 在相同 model+date+sha 路径已存在时一律跳过张量上传(不区分 pass_label),随后通过的重跑只是追加 `passed` 行,于是该行复用了失败 run 留在原路径的偏差张量并成为下一段 baseline。

## 修复方案

- `python/sglang/test/precision_baseline_store.py` 的 `push_run`:仅对 `failed` push 跳过张量上传;passed push 在已存在路径时上传自己的张量,保证 `passed` 行永远指向通过 run 的张量;failed push 仍保留原张量不动。

```diff
-    skip_tensors = tensors_already_present and not force
+    skip_tensors = (
+        tensors_already_present and not force and meta["pass_label"] == "failed"
+    )
```

- CI context 新增记录 `RUNNER_NAME`(存为 `ci_runner`)并提升到 manifest,使跟着某台机器走的偏差可从 manifest 识别。
- 当前 store 里已有的坏 baseline 仍需健康 runner 用 `SGLANG_PRECISION_FORCE_UPDATE=1` 刷新一次;本 PR 本身不写 store,也未重跑该 nightly 测试(它会写共享 store)。

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 `precision_baseline_store.py` 的 `push_run` 上传去重逻辑与 nightly 精度测试的 CI context 记录;failed push 的跳过行为不变。作者已用假 `HfApi` 本地验证:失败后同路径的 passed push 会在此处上传自己的张量、在 main 上跳过;failed push 仍跳过。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
