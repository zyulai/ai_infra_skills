# PR #40598: [AMD][Fix] AgentX HIP TPOT regression when SGLANG_SIMULATE_ACC_LEN is set

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#40598](https://github.com/sgl-project/sglang/pull/40598) |
| 作者 | yichiche |
| 合并日期 | 2026-09-22 |
| labels | amd, speculative-decoding, run-ci |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
HIP 上 #37134 自动开启 rejection sampling,导致 AgentX 模拟接受路径白跑 Triton chain sampler、TPOT 恶化约 14%;在模拟接受场景跳过自动开启即恢复。

## 问题现象
Agent 模式(InferenceMax AgentX)对 Qwen3.5 设置 `SGLANG_SIMULATE_ACC_LEN=3.39`(`match-expected` / `real-draft-token`):服务端仍跑完整 EAGLE verify,然后覆写接受长度。#37134 在 HIP 上自动启用 `--speculative-use-rejection-sampling` 后,每个 decode 步还额外跑 Triton chain sampler,而其结果被丢弃。实测 AgentX TPOT(TP4 conc=4)较 pre-#37134 / 0915 baseline 恶化约 14%。基准:InferenceX arm `qwen3.5-fp4-mi355x-sglang-agentic-mtp`,TP4、conc=4、`DURATION=1200`,容器 `rocm/sgl-dev:v0.5.19-rocm720-mi355x-20260915`;`832ec39cc`(pre-#37134)TPOT 3.1ms / 317.9 tok/s/user,`8d08dfdab`(含 #37134)TPOT 3.6ms / 277.5 tok/s/user。首恶提交为 #37134(`7eedd57ab`),父提交 `2f7d04da5` 正常。定长 MTP 不设置 `SGLANG_SIMULATE_ACC_LEN` 且客户端贪心(`temperature=0`),verify 走 `argmax`,不受影响。

## 根因
正文披露:#37134 在 HIP 上自动启用 rejection sampling,而 AgentX(`SGLANG_SIMULATE_ACC_LEN` 设置时)仍做真实 EAGLE verify 后覆写接受长度,chain sampler 的结果被白白计算丢弃,多出的 verify 开销体现为 TPOT 回归。即自动开启的门控未排除模拟接受长度场景。

## 修复方案
改 `python/sglang/srt/arg_groups/speculative_hook.py`:
- `_should_auto_enable_hip_rejection_sampling` 新增参数 `simulate_acc_len: float = -1.0`,并在门控条件中增加 `and simulate_acc_len <= 0`——当 `SGLANG_SIMULATE_ACC_LEN > 0` 时跳过 HIP 自动开启;真实评测/服务(env 未设置或 `< 0`)仍按 #37134 自动开启。
- `_handle_eagle_family` 调用门控时传入 `float(envs.SGLANG_SIMULATE_ACC_LEN.get())`。
- 显式指定 `--speculative-use-rejection-sampling` 的行为不变(helper 在 flag 已开时本就 no-op)。
- 新增单测 `test_simulate_acc_len_stays_off`(`test/registered/unit/spec/test_eagle_gate_routing.py`),覆盖 `simulate_acc_len=3.39` 返回 False、`-1.0` 返回 True。

关键 diff 摘录:
```diff
+        and not enable_deterministic_inference
+        and simulate_acc_len <= 0
     )
```

## 影响与回归风险
- 影响窗口:含 #37134(`7eedd57ab`)之后到本修复之前的提交,AgentX HIP 路径(TP4 conc=4 下 TPOT 3.1→3.6ms);正文明确给出首恶提交,父提交 `2f7d04da5` 正常。
- 回归面:仅 HIP 自动开启 rejection sampling 的判定(`_should_auto_enable_hip_rejection_sampling` 及其唯一调用点 `_handle_eagle_family`)。修复后 patch 基准恢复至 TPOT 3.2ms / 315.6 tok/s/user / 21158.2 tok/s、350/357 ok,与 pre-#37134 基线相当;`EVAL_ONLY=true` AgentX 路径不设置 `SGLANG_SIMULATE_ACC_LEN`,行为不变;非 HIP 平台与显式 flag 不受影响。

## 关联
- [#37134](https://github.com/sgl-project/sglang/pull/37134)(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
