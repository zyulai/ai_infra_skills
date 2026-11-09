# PR #44993: [Bugfix][Structured Output][Spec Decode] Advance grammar across reasoning boundary

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44993](https://github.com/vllm-project/vllm/pull/44993) |
| 作者 | yuyue0225sc |
| 合并日期 | 2026-07-23 |
| labels | bug, structured-output, speculative-decoding, ready, v1 |
| 热度 | 👍 7 · 💬 39 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
async+spec decode+reasoning parser 下 grammar 跨不过 reasoning 边界(delta 窗口漏标记、标记后 token 不进 FSM);统一 should_advance 边界路径修复。

## 问题现象
async scheduling + speculative decoding + Qwen 风格 reasoning parser 组合下,`should_advance` 对 JSON/regex/choice 约束静默失效:开启 `response_format=json_object` 时输出带 Markdown 代码围栏或任意前导文本;甚至返回 `{{...}` 这类双花括号内容——HTTP 200 但客户端解析失败(#48228 报告的原始症状)。基线实测 Qwen3.5-35B-A3B + MTP k=4 场景 50 次请求仅 7 次返回干净 JSON(86% 失败)。

## 根因
正文披露两个相互作用的 bug:
1. **delta 窗口漏掉标记**:delta 窗口由 `num_computed_tokens - num_output_placeholders` 计算,当部分 draft 被拒绝时 placeholder 数保持大于 0,窗口起点越过 `</think>` 标记,`reasoning_ended` 永不翻转,grammar 从未生效。
2. **标记后的 token 进不了 FSM**:即使 `reasoning_ended` 翻转,标记同一步产生的标记后内容 token 也从未进入 grammar FSM;下一步 bitmask 用 grammar 初始状态准备,模型重新输出开括号,于是产生 `{{...}`。

## 修复方案
统一所有约束类型的边界路径:
- `should_advance`(vllm/v1/structured_output/__init__.py)新增可选 `new_token_ids` 参数,调度器调用点传入本步追加的 token,直接作为精确 delta 窗口(修 Bug 1);两个 draft 校验调用点保持 placeholder 回退不变。
- 边界处记录 `reasoning_end_token_index` 并对**所有**约束类型返回 `True`;调度器(vllm/v1/core/sched/scheduler.py 的 `update_from_output`)用 `trim_reasoning_for_advance()` + `accept_tokens()` 只推进已验证的标记后后缀(修 Bug 2)。`should_advance` 不再改动 grammar、不再有 STRUCTURAL_TAG 特例分支。
- 语义变更:边界处 `accept_tokens` 被拒(正常流不应发生)现在走调度器标准错误路径终止请求,而非 warn-and-continue。
- 新增 3 个回归测试。同类的 intra-step bitmask 泄漏由 #44297 修复,两处补丁点独立。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;#43388 在 Qwen3.5 + MTP 上仍可复现,#36138(已合并)的 placeholder 数学在 async + draft 拒绝下下溢是诱因之一。
- 回归面:`should_advance` 的全部调用点(调度器 + 两处 draft 校验)与 reasoning→content 边界的 FSM 推进时序;验证覆盖 Qwen(MTP)、gemma-4-E2B(MTP)、DeepSeek-V4-Flash-DSpark(DSpark)与 STRUCTURAL_TAG 路径,确认未回退 #44297。

## 关联
Fixes #43388, #48228, #34650;相关 #44297, #36138, #43526, #43424, #44006

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
