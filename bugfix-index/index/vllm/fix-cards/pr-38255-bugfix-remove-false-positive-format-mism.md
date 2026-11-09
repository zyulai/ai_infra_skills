# PR #38255: [Bugfix] Remove false-positive format mismatch warnings in FLA ops

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38255](https://github.com/vllm-project/vllm/pull/38255) |
| 作者 | tdoublep |
| 合并日期 | 2026-03-30 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
FLA 算子的 "potential format mismatch" 告警在 seq_len < num_heads 时误报(chunked prefill/prefix caching 下属正常),删除该启发式告警修复。

## 问题现象
`chunk_gated_delta_rule` 与 `chunk_local_cumsum` 两个 FLA 算子在推理时不断发出 "Input tensor shape suggests potential format mismatch" 警告。触发条件是 `seq_len < num_heads`,而这在 vLLM 使用 chunked prefill 或 prefix caching 的推理中是正常状态(例如 16-token 的 chunk 配 32 个 head)。告警是误报(false-positive),会刷屏干扰用户;对 Qwen3.5 等使用 FLA/GDN 的模型推理均会出现。

## 根因
PR 正文披露:该告警基于 `seq_len < num_heads` 的形状启发式来判断输入是否被误传成 head-first 格式 `[B, H, T, ...]`,但这一启发式完全没有考虑短序列的情况——张量实际上始终是正确的 `[B, T, H, ...]` 格式,因此短 chunk 场景下必然误报。

## 修复方案
删除两处告警逻辑及不再使用的 `import warnings`:
1. `vllm/model_executor/layers/fla/ops/chunk.py` 的 `chunk_gated_delta_rule`:删掉 `if q.shape[1] < q.shape[2]: warnings.warn(...)` 整块;
2. `vllm/model_executor/layers/fla/ops/cumsum.py` 的 `chunk_local_cumsum`:删掉 `if not head_first and g.shape[1] < g.shape[2]: warnings.warn(...)` 整块。

```diff
-    if q.shape[1] < q.shape[2]:
-        warnings.warn(
-            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
-            ...
-        )
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:纯删除诊断性 warning,不改变任何数值计算路径,FLA/GDN(如 Qwen3.5)推理输出不受影响;代价是若调用方将来真的把 head-first `[B, H, T, ...]` 格式误传进来,不再有任何格式告警提示。

## 关联
- Issue #37103(本 PR 声明 Fixes 该 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
