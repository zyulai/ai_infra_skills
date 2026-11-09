# PR #31675: [AMD] Fix DeepSeek MLA prefill shape mismatch on HIP eager fallback (missing mha_companion_layers)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31675](https://github.com/sgl-project/sglang/pull/31675) |
| 作者 | yctseng0211 |
| 合并日期 | 2026-07-19 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

HIP eager fallback 漏传 mha_companion_layers,MHA q 用错误 qk_head_dim reshape 报错;补传参数修复。

## 问题现象

AMD (HIP) 上 DeepSeek MLA prefill 在 prefill runner 使用 `tc_piecewise` 时崩溃,报错:

```
RuntimeError: shape '[-1, 16, 576]' is invalid for input of size 36003840
```

失败运行记录:https://github.com/sgl-project/sglang/actions/runs/29644059344/job/88079191396 。触发条件是 DeepSeek MLA prefill 走 HIP PCG eager-fallback 路径。

## 根因

正文披露:PR #31391 将逐层的 `_pcg_mha_companion` 属性(用于把 `attn_mqa` 换成 `attn_mha`,使 attention backend 在 MHA 路径上看到正确的 head/dim 元数据,`save_kv_cache=False`)改为经 context 传递的 `mha_companion_layers` 列表。该 PR 更新了 `prefill_cuda_graph_runner.py` 中三处 `set_tc_piecewise_forward_context(...)` 调用点,但遗漏了 `eager_runner.py` 中 HIP PCG eager-fallback 的调用点。

因此在 AMD eager-fallback 路径上 `context.mha_companion_layers` 为 `None`,`radix_attention.py` 的 `unified_attention_with_output` 不会切换到 `attn_mha`;backend 随后用 `attn_mqa` 的 `qk_head_dim`(kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576)去 reshape MHA q(head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192),view 失败(36003840 = 16 * 192 * 11720 不能被 16 * 576 整除)。

## 修复方案

在 `python/sglang/srt/model_executor/runner/eager_runner.py` 的 HIP eager-fallback 路径中,向 `set_tc_piecewise_forward_context` 传入 `mha_companion_layers=model_runner.mha_companion_layers`,与 `prefill_cuda_graph_runner.py` 已修复的三处调用点保持一致。正文说明 `model_runner.mha_companion_layers` 在 `capture_prefill_graph` 中与 `attention_layers`/`moe_layers`/`dsa_indexers` 一同填充,且该分支由 `pcg_runner is not None` 守卫,属性在此必然可用。

关键 diff 摘录:

```diff
                         model_runner.moe_layers,
                         model_runner.moe_fusions,
                         dsa_indexers=model_runner.dsa_indexers,
+                        mha_companion_layers=model_runner.mha_companion_layers,
                     ):
```

## 影响与回归风险

- 影响窗口:自 PR #31391 引入 `mha_companion_layers` context 传递机制起,至本修复前;正文以 CI 失败运行链接佐证。
- 回归面:仅 `eager_runner.py` 的 `_execute_extend` 中 set_tc_piecewise_forward_context 调用新增一个参数;使 HIP eager fallback 与 CUDA graph runner 行为一致。影响 AMD DeepSeek MLA prefill 的 MHA companion 路径(attn_mqa -> attn_mha 切换),不触及 CUDA graph 路径。正文未提供 accuracy 测试结果(Checklist 均未勾选,CI 状态栏显示 PR Test 为失败状态)。

## 关联

- PR #31391(正文 Root cause 提及,引入回归的 PR)
- 失败 CI 运行: https://github.com/sgl-project/sglang/actions/runs/29644059344/job/88079191396

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
