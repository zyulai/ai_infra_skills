# PR #5830: [PD] Fix Assertion failed: /DeepEP/csrc/kernels/internode.cu:483, condition: ibgda_get_state()->num_rc_per_pe >= num_channels #134

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5830](https://github.com/sgl-project/sglang/pull/5830) |
| 作者 | whybeyoung |
| 合并日期 | 2025-04-30 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
PD 场景 DeepEP internode kernel 断言 `num_rc_per_pe >= num_channels` 失败而崩溃;`get_deepep_buffer` 的 `num_qps_per_rank` 改为 `max(num_experts // group.size(), Buffer.num_sms // 2)` 保证下限。

## 问题现象
正文未描述具体现象,Motivation 只写「Fix https://github.com/deepseek-ai/DeepEP/issues/134」。从标题看,崩溃表现为 DeepEP CUDA kernel 内断言失败:

```
Assertion failed: /DeepEP/csrc/kernels/internode.cu:483,
condition: ibgda_get_state()->num_rc_per_pe >= num_channels
```

标题带 [PD] 前缀,涉及 PD(prefill-decode disaggregation)部署中 DeepEP internode 通信路径。

## 根因
正文未披露根因。(由 diff 推断)`get_deepep_buffer` 创建 DeepEP `Buffer` 时,`num_qps_per_rank` 在 low-latency 模式下取 `num_experts // group.size()`、其余情况取 1;当该值小于 internode kernel 侧要求的通道数(断言中的 `num_channels`)时,`ibgda` 初始化出的每 PE RC 数量不足,断言失败、进程崩溃。修复按 `Buffer.num_sms // 2` 给出下限,确保 QP 数不低于 kernel 期望。

## 修复方案
`python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` 的 `get_deepep_buffer`:

```diff
-            num_qps_per_rank=(
-                num_experts // group.size() if deepep_mode.enable_low_latency() else 1
-            ),
+            num_qps_per_rank=(max(num_experts // group.size(), Buffer.num_sms // 2)),
```

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:EP MoE DeepEP token dispatcher 的全局 buffer 初始化,所有经 `get_deepep_buffer` 创建 buffer 的路径。行为变化:`num_qps_per_rank` 从原来的条件表达式(low-latency 时 experts/rank,否则 1)统一变为 `max(num_experts // group.size(), Buffer.num_sms // 2)`——包括非 low-latency 模式,其 QP 数不再固定为 1。正文未提供验证或测试结果。

## 关联
- [deepseek-ai/DeepEP#134](https://github.com/deepseek-ai/DeepEP/issues/134)(本 PR 修复的上游 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
