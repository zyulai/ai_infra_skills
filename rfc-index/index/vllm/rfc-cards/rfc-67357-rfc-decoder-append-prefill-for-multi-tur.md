# RFC #67357: [RFC]: Decoder Append-Prefill for Multi-Turn Workloads in P/D Disaggregated Serving

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#67357](https://github.com/vllm-project/vllm/issues/67357) |
| 作者 | Ydr-pku |
| 创建日期 | 2026-11-13 |
| 状态 | 🟢 open |
| labels | RFC, kv-connector |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议 opt-in 的 Decoder Append-Prefill:多轮会话的后续请求复用上一轮 Decoder 的本地 KV 并在 Decoder 本地补齐剩余 prefill,本地缓存不可用时回落普通 P/D,以缓解 Prefiller 压力。

## 提案动机
多轮 agentic workload 的会话历史长且持续增长、输出相对较短;在 P/D 分离服务下,这类负载会把压力集中到 Prefiller 实例,请求突发时尤甚。上一轮服务该会话的 Decoder 本地 KV cache 中可能仍保留大量可复用前缀(包括该轮生成的 token);把下一请求照常送 Prefiller 会引入排队、额外 prefill 计算与 KV 传输——即使 Decoder 本可复用本地前缀、只计算剩余 suffix。目标负载:Prefiller 压力大、Decoder 有富余容量、未缓存 suffix 相对较短。设计必须顾及追加的 prefill 工作对进行中 decode 请求的影响。

## 提出的方案
1. **请求流**:首轮走普通 P/D;router 记录服务该 session 的 Decoder endpoint 与 data-parallel rank;对 eligible 的后续轮,router 把完整请求直接发给该 Decoder/rank 并请求 local-cache admission;Scheduler 检查实际可复用的本地 KV 前缀,获准则由 Decoder 计算未缓存 suffix 并生成响应;条件不满足时,Decoder 在执行 prefill 或生成输出之前拒绝本次尝试,router 改走普通 P/D 重试原请求。客户端继续提供完整会话历史;可复用范围由实际 token-prefix match 决定;session affinity 只用于定位可能的 cache owner,不保留 KV 块、也不保证 residency。
2. **请求级 admission 策略**:引入一个请求级选项,表达「本地 prefill 以存在可用的本地 KV 前缀为条件」。参考实现用 `require_decoder_cache_hit`,可用本地前缀非零即 admit 新的本地尝试;upstream 字段名与 API 位置开放讨论。仅非零 hit 并不约束剩余 prefill 工作量,作者欢迎反馈初始接口是否还应支持最小 hit 阈值或最大 uncached-suffix budget;外部 router 也可按自身路由与负载策略决定是否尝试本地执行。该功能应默认禁用。
3. **结构化拒绝与 Scheduler 集成**:引擎需要一个机器可读结果,把 admission 拒绝与成功生成及其他失败区分开;对流式请求,zero-token rejection 必须可靠到达 router,router 应在内部消化被拒尝试并保留原请求与生成预算用于 P/D 回退;admission 完成前发出的协议 metadata 不得让被拒尝试看起来像成功响应。建议把 admission 集成进现有 Scheduler cache-lookup 路径,允许拒绝单个 waiting request 并继续调度其他请求,同时保持 KV 块生命周期、请求清理与同步/异步输出顺序。参考实现目前靠复制 upstream 调度主体来插入该检查,upstream admission 点可减轻维护负担;征求反馈:做成直接的 Scheduler policy 还是小型扩展接口。
4. **抢占与 KV 恢复**:新 follow-up 请求的 admission 与既有被抢占请求的恢复须分开处理:有有效 offload recovery state 的既有请求应可经 connector lifecycle 恢复自己的 KV;而早前请求的 CPU offload 副本本身并不构成新 turn 的 local-cache admission。零输出拒绝可经普通 P/D 回退;输出已生成后的恢复失败需要显式 terminal result(除非存在准确的 continuation 协议)。经 Prefiller 重建做透明续写属于另一项工作,需保留原始输入 token、已生成 token、剩余生成预算与流式语义;本 RFC 不假设把生成文本回放进 chat messages 就能提供该保证。
5. **性能与验证**:评估需同时覆盖收益与 Decoder 干扰:E2E 延迟、TTFT、吞吐与 Prefiller 排队;并发 decode 请求的 TPOT 与尾延迟;实际 local-cache admission 率与 cache-miss 回退成本;完整多角色会话历史下的正确性;cache 驱逐、DP 路由、取消、同步/异步调度;支持处的 same-request offload 恢复。负载应变化会话长度、uncached-suffix 长度、到达率、Decoder 压力与 cache residency。

## 优化点与收益
本 RFC 正文未给出自身的量化收益数字;收益定位是在 Prefiller 高压、Decoder 有余量、uncached suffix 较短的负载下,减少 Prefiller 排队、重复 prefill 计算与 KV 传输。参考实现在 vLLM-Ascend PR #19884 附带部署相关评估:MRV2 benchmark 方法、local admissions 与 P/D fallback 统计、TPOT trade-off,以及单独的 same-request offload-restoration 验证;作者明言这些结果只覆盖所测的 Ascend 部署,其他 backend 的验证仍待进行。

## 现状与落地
open(2026-11-13 创建,👍 0 · 💬 0);upstream 尚无实现,参考实现位于 vLLM-Ascend PR #19884。向社区提出 5 个问题:请求级 local-cache admission 是否合适的 upstream 抽象、admission 拒绝应如何经 engine outputs 与 serving APIs 表达、初始策略是只要求可用前缀还是同时限制 uncached-suffix 工作量、最小可维护的 Scheduler 集成形态是什么、初始验证范围应包含哪些 connector 与硬件配置。

## 关联
- [vLLM-Ascend PR #19884](https://github.com/vllm-project/vllm-ascend/pull/19884)(参考实现与 Ascend 部署评估)
- [RFC #24256: Cache-hit-based admission control](https://github.com/vllm-project/vllm/issues/24256)(本提案是其 local-cache 方向的用例;该早期 RFC 还考虑 external/offloaded cache 与 preemption 处理,作者希望在其讨论基础上定义 admission 接口)
- [RFC #32733: Bidirectional P/D KV transfers](https://github.com/vllm-project/vllm/issues/32733)(提议把 Decoder 可复用 KV 传给 Prefiller,并明确讨论在 Decoder 上做 incremental prefill 作为替代;本 RFC 探索用 Decoder 算力为合适请求绕过 Prefiller 的取舍,其价值取决于负载与 Decoder 可用容量)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
