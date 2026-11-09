# RFC #47797: [RFC] Define per-request speculative decoding opt-out execution semantics

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47797](https://github.com/sgl-project/sglang/issues/47797) |
| 作者 | anencore94 |
| 创建日期 | 2026-11-01 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
重启 #30263 的设计讨论:为逐请求投机解码 opt-out 定义可执行的语义与验收标准,而非只传一个不生效的 flag。

## 提案动机
#30263 与 draft #30264 已被 inactivity 自动化关闭,GitHub 拒绝重开原 issue。当前 main(a52ee805)仍没有 request 级的 disable_speculative_decoding 字段或消费者,逐请求投机统计(含 #34814)也不提供 opt-out。作者强调:原 draft 只传递了一个不强制执行的 flag,不应在没有实现其执行语义的情况下把该 API 恢复成"可用功能"。

## 提出的方案
- 方案一:把 opt-out 请求分组进非 speculative 的 scheduler batch,跨 batch 切换保持 ordering、streaming 与 KV-cache ownership——使混合请求行为显式,但增加调度与执行路径复杂度。
- 方案二:对含 opt-out 请求的 batch 使用显式的 steps-zero runtime state,可复用 adaptive speculative 的运行时机制,但需定义混合 batch 策略并正确匹配 attention/CUDA-graph 状态;只改请求 flag 或在 worker 内部分支都不够。
- 现有线索:EAGLEWorkerV2 已有 speculative_num_steps == 0 路径,adaptive_spec_params.py 管理运行时切换——这些是调查点,而非逐请求开关安全的证明。
- 合并 PR 前的验收标准:opt-out 请求不执行任何 draft 模型工作,且确定性采样下与非 spec target 输出一致;默认请求保持既有行为,并明确定义两类请求同 batch 时的行为;测试 eager 与 CUDA-graph 执行、streaming/cancellation、batch 切换与 KV-cache 生命周期;在低/高并发下测 latency/throughput 并记录调度取舍;不接受被静默忽略的公开参数。

## 优化点与收益
正文未给出量化数字。本 RFC 聚焦语义正确性:确保公开参数要么按定义生效、要么不提供,消除"静默忽略"。

## 现状与落地
open。作者表示一旦混合 batch 契约与首选方案达成一致即恢复实现,并向社区征询:两种方案哪种更适合当前 scheduler/runtime 架构。

## 关联
- #30263(前次逐请求 opt-out RFC,已关闭)
- Draft PR #30264(原 API plumbing)
- #34814(逐请求投机统计)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
