# PR #12900: Revert "Fix spec decoding acc length for dpsk-r1-fp4 tp8"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12900](https://github.com/sgl-project/sglang/pull/12900) |
| 作者 | Qiaolin-Yu |
| 合并日期 | 2025-11-09 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
回退 #12896:它把 flashinfer 通信融合工作区默认 max_token_num 从 16384 降到 2048,撤销该改动。

## 问题现象
正文未描述具体现象,仅说明回退 sgl-project/sglang#12896。

## 根因
正文未披露根因。(由 diff 推断)被回退的 #12896 把 `flashinfer_comm_fusion.py` 中 `ensure_workspace_initialized` 与 `flashinfer_allreduce_residual_rmsnorm` 的默认参数 `max_token_num` 从 16384 改为 2048,即把 flashinfer allreduce+rmsnorm 融合的工作区默认容量缩小到 1/8;本 PR 恢复 16384。原「修复」针对 dpsk-r1-fp4 tp8 的 spec decoding acc length 问题,推测缩小工作区引发了新的问题,故整体撤销。

## 修复方案
整体 revert #12896,`python/sglang/srt/layers/flashinfer_comm_fusion.py` 两处默认参数恢复:

```diff
-    max_token_num: int = 2048, hidden_dim: int = 4096, use_fp32_lamport: bool = False
+    max_token_num: int = 16384, hidden_dim: int = 4096, use_fp32_lamport: bool = False
```

`flashinfer_allreduce_residual_rmsnorm` 的默认 `max_token_num` 同样从 2048 恢复为 16384。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:使用 flashinfer allreduce+rmsnorm 融合路径(如 TP 内通信融合)的工作区大小默认值;#12896 想解决的 spec decoding acc length 问题可能随之重现,需另行处理。

## 关联
- sgl-project/sglang#12896(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
