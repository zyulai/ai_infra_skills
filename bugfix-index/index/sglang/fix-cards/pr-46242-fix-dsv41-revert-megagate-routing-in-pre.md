# PR #46242: [Fix] dsv41: Revert MegaGate routing in prefill

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46242](https://github.com/sgl-project/sglang/pull/46242) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-10-24 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #46232:prefill 重新启用 MegaGate 后 GB300 上第二个 prefill graph 捕获报 unspecified launch failure、服务器无法启动,恢复 decode/verify 门控与 2048 行上限。

## 问题现象
`SGLANG_OPT_DEEPGEMM_MEGA_GATE=1` 且 #46232 生效时,GB300(DeepSeek-V4.1-Flash,DP attention 4 + MegaMoE)上 4096 与 8192 行两个 prefill graph 形状中,第二个形状的捕获失败,非零 DP rank 报 `CUDA error: unspecified launch failure`,服务器无法启动;用 `sgl-deep-gemm` 0.2.3.post2 可复现。这一捕获失败本已被 #45969 修掉,#46232 使其复发。

## 根因
正文披露:#46232 让 MegaGate 重新跑在 extend batch 上,包括可中断(breakable)的 prefill CUDA graph 内部,带回了 #45969 修过的 prefill 捕获失败。#46232 的动机是恢复 prefill MegaGate,但捕获失败机制尚未弄清。

## 修复方案
恢复 decode / target-verify 门控与 2048 行硬上限,与 #46232 之前完全一致:

```diff
-    max_rows = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
     return (
-        (mode.is_decode() or mode.is_extend())
+        (mode.is_decode() or mode.is_target_verify())
         and moe.mega_gate_supported
-        and 0 < hidden_states.shape[0] <= max_rows
+        and 0 < hidden_states.shape[0] <= _MEGA_GATE_MAX_ROWS
```

并恢复常量 `_MEGA_GATE_MAX_ROWS = 2048`;单测恢复 prefill(EXTEND)不启用、TARGET_VERIFY 启用、4096 行超限不启用的断言,并覆盖「prefill graph capture 会清除 is_extend_in_batch,须按 forward mode 本身门控」的场景。验证:`mega_gate.py` 与 `test_mega_moe_deepgemm_api.py` 与 #46232 的父提交字节一致,MegaGate admission 测试通过。正文明确:prefill MegaGate 待捕获失败被理解后可再引入。

## 影响与回归风险
- 影响窗口:#46232 合并至本回滚之间。
- 回归面:MegaGate 准入逻辑回到 prefill 不启用的状态;#46232 带回的 prefill MegaGate 能力(含行数上限改由环境变量控制)被撤回,依赖该能力的 prefill 路径回到常规 router。

## 关联
- [#46232](https://github.com/sgl-project/sglang/pull/46232)(被回滚的 PR)
- [#45969](https://github.com/sgl-project/sglang/pull/45969)(当初修复 prefill 捕获失败的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
