# RFC #62818: [RFC]: Preserve output distribution under structured-output + speculative decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62818](https://github.com/vllm-project/vllm/issues/62818) |
| 作者 | yzong-rh |
| 创建日期 | 2026-10-21 |
| 状态 | ⚪ closed(关闭于 2026-10-27) |
| labels | RFC, speculative-decoding, structured-output |
| 热度 | 👍 1 · 💬 14 |
| 分类 | Speculative Decoding |

## 一句话结论
修复结构化输出与投机解码叠加时输出分布被改变的问题:保留首个被拒草稿 token 交给 rejection sampler,使其按残差分布重采样。

## 提案动机
结构化输出 + 投机解码下,`StructuredOutputManager` 分三步处理草稿 token:`validate_tokens` 不提交语法状态地检查草稿序列并裁剪为语法合法前缀;`grammar_bitmask` 为每个草稿 + bonus 位置构建词表掩码;`accept_tokens` 把验证通过的 token 提交进持久语法状态。问题(见 #60451)在于当前生命周期会改变输出分布:假设草稿为 `[valid, invalid, ?, ?]`,`validate_tokens` 会裁剪成 `[valid]`,于是 rejection sampling 正确采样 `valid` 后,把 `invalid` 当作 bonus token 而不是被拒绝的 token——该位置会从 target(验证模型)分布采样,而不是从残差分布 `max(target - draft, 0)` 采样。异步调度下该问题更结构性:step N 是用未验证的 placeholder 草稿调度的,同步调度下则是用已验证草稿调度。

## 提出的方案
- 改动 1(核心):保留首个被拒绝的草稿给 rejection sampler——`validate_tokens` 把 `[valid, invalid, ?, ?]` 裁剪为 `[valid, invalid]` 而非 `[valid]`;由于 grammar bitmask 不允许 `invalid`,rejection sampler 会把该位置当作拒绝并重采样,而不是当作 bonus。代价:同步调度下 target model 要多验证一个 token,带来小幅性能回归。
- 改动 2(可选):异步调度路径不再调用 `validate_tokens` 裁剪草稿——请求本就带着未验证 token 被调度,该调用是冗余的;移除它同时去掉 reasoning-end 检测与一次临时语法状态前进/回滚。
- 实现 PR #61557:将 `validate_tokens` 改名为 `trim_spec_tokens` 并确保 invalid token 不被截断(改动 1);移除异步路径的调用(改动 2);修正 grammar backend 保证 `accept_tokens` 保持事务性——避免在 `grammar_bitmask` 内部多调一次 `validate_tokens`。
- 备选 PR #60690:用 `SchedulerOutput.num_invalid_spec_tokens` 保证 invalid token 不被丢弃,并为同步路径在 `Request` 上记录该计数,使 step N 计算的结果在 step N+1 可用。
- 被否决的方案:在同步与异步路径都不做 `validate_tokens`——技术上能同时实现两个改动,但会在同步调度路径上造成统计显著的性能回归。

## 优化点与收益
正确性修复,正文未给出量化数字。收益:结构化输出 + 投机解码叠加时输出分布恢复与纯 rejection sampling 一致的语义(被拒 token 从残差分布 `max(target - draft, 0)` 采样而非 target 分布);可选改动 2 还省去异步路径的冗余验证与一次语法状态前进/回滚;明确接受的代价是同步路径多验证一个 token 的小幅回归。

## 现状与落地
已关闭(2026-10-27)。正文给出实现 PR(#61557)与备选 PR(#60690),并说明被否决的替代方案及理由;最终合入情况需查关联 PR。

## 关联
- 关联 issue: https://github.com/vllm-project/vllm/issues/60451
- 实现 PR: https://github.com/vllm-project/vllm/pull/61557
- 备选 PR: https://github.com/vllm-project/vllm/pull/60690

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
