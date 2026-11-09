# RFC #48296: [RFC] Speculative Decoding Lifecycle for RL Rollout

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48296](https://github.com/vllm-project/vllm/issues/48296) |
| 作者 | aoshen02 |
| 创建日期 | 2026-07-11 |
| 状态 | ⚪ closed(关闭于 2026-07-13) |
| labels | RFC, rl |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
界定 RL rollout 使用 vLLM 推测解码(MTP/EAGLE/suffix)被卡住的三个生命周期缺口,并定义可验证的退出标准。

## 提案动机
RL 框架无法用 vLLM 的推测解码做 rollout 加速,原因有三:①配置未暴露 —— verl 的 `vllm.LLM` wrapper 不透传 `speculative_config`(verl#4526);②MTP 权重同步损坏 —— RL 训练更新 MTP head 权重后,`load_weights` 不会重载 MTP 参数,权重过期(verl#6947);③draft 模型生命周期 —— EAGLE/EAGLE3 draft 模型在 sleep/wake 期间需要权重重载,当前只更新 verifier 权重(#46725,已在第 1.1 节追踪)。

## 提出的方案
- 汇总下游需求佐证:verl#5925(EAGLE/EAGLE3 for RL rollout,PR open)、verl#6342(suffix spec decode in vLLM rollout,PR open)、verl#4791(suffix + LSTM spec decode RFC)、SkyRL#1832(MTP 训练支持,SkyRL Q3 路线图 P0)。
- 明确退出标准(exit criterion):
  1. verl 能为 vLLM rollout 配置 `speculative_config` 而不报错;
  2. MTP head 权重在 `load_weights` / `finish_weight_update` 后正确重载。
- 反馈期:发布起两周。
正文属于问题界定与协调类 RFC,未给出实现设计或代码方案。

## 优化点与收益
正文未给出量化数字;收益为打通 RL rollout 侧推测解码的可用性(config 透传 + 权重同步正确性),支撑 verl、SkyRL 等框架已排队的需求。

## 现状与落地
已关闭 —— 发布两天后(2026-07-13)关闭,0 条评论;正文未写明是否实现或由哪个 PR 承接,落地情况需查关联 PR。

## 关联
- [verl#4526](https://github.com/verl-project/verl/issues/4526) 与 [verl#6947](https://github.com/verl-project/verl/issues/6947)(配置未透传、MTP 权重不重载)
- [vLLM PR #46725](https://github.com/vllm-project/vllm/pull/46725)(sleep/wake 权重相关)
- [verl#5925](https://github.com/verl-project/verl/pull/5925) / [verl#6342](https://github.com/verl-project/verl/pull/6342) / [verl#4791](https://github.com/verl-project/verl/issues/4791)(下游需求)
- [SkyRL#1832](https://github.com/NovaSky-AI/SkyRL/pull/1832)(SkyRL Q3 路线图 P0)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
