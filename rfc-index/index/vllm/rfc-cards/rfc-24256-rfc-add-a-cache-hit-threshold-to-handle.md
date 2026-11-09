# RFC #24256: [RFC]: Add a cache hit threshold to handle Preemptions in PD-Disaggregation and enable lightweight powerful P/D implementations

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24256](https://github.com/vllm-project/vllm/issues/24256) |
| 作者 | kfirwolfson |
| 创建日期 | 2025-09-04 |
| 状态 | ⚪ closed(关闭于 2026-05-31) |
| labels | RFC, stale |
| 热度 | 👍 5 · 💬 5 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议为请求增加可选的 KV cache 命中率阈值,低于阈值即拒绝,把 P/D 控制权交还外部 Router。

## 提案动机
面向 PD-Disaggregation 与抢占处理场景。1) **请求抢占**:vLLM 抢占时会清空请求的 KV cache 块,被抢占请求稍后在 Decode 实例内部重做完整 Prefill(包括原本该在 Prefiller 做的部分与抢占前已算出的输出 token);llm-d 团队的实测显示,该场景会导致 Decode 实例开始执行 prefill 并因资源耗尽最终锁死,而外部 Router 在请求进入 Decode 实例后无法干预 vLLM 行为。长输出 workload 更易触发(生成 token 越多、KV cache 越容易耗尽)。2) **轻量强大的 P/D**:一些 PD-D 系统依赖带全局索引的 cache-aware Router(靠 KV Events 持续更新),规模到 PB 级存储后复杂且昂贵;更好的做法是把查找放在存储层并由 vLLM 经 KVConnector 的 `get_num_new_matched_tokens()` 完成,免去 Router 访问存储、减少每请求查找次数。此外该特性还可实现简单的 cache-aware 路由:Router 并发给 K 个实例、只有真有缓存的实例处理请求。

## 提出的方案
- 新全局配置 `--global-cache-hit-threshold`(float ∈ [0.0, 1.0],默认 0.0)与新 per-request 字段 `cache_hit_threshold`(float ∈ [0.0, 1.0]),请求级覆盖全局,兼顾静态配置与 Router 动态覆盖。
- 命中率计算:scheduler 在调用 `get_num_new_matched_tokens()` 之后、请求调度与 cache block 分配之前计算,保证轻量拒绝;hit rate =(local + external)cached tokens / prompt length,外部与先前卸载的 cache 经 KVConnector 计入,不只看本地 APC。
- 低于阈值即拒绝。拒绝方式 Option A(首选):仍返回 HTTP 200,引入新 finish_reason `cache_threshold`,与现有 `stop`/`length`/`abort` 语义一致,可携带部分 token 数据;Option B(备选):返回 HTTP 422 错误对象,但与「返回已算出的输出 token」的抢占优化不兼容。
- 抢占优化(phase 2,可选):拒绝时 vLLM 把已生成的 output tokens 一并返回;Router 克隆新请求——prompt 追加这些 token,并从 `max_tokens`/`max_completion_tokens` 中扣除——交由 Prefiller 补做 prefill,避免 Decoder 重算。
- 兼容性:默认阈值 0.0,存量部署行为不变;实现预计只涉及 Scheduler 与请求解析层,无向后不兼容的接口/schema 变更。

## 优化点与收益
收益为系统可靠性(避免 Decoder 因抢占后重 prefill 而锁死)与架构简化(Decode-First 场景把是否 P/D 的决策下沉到 vLLM 内一次查找)。正文引用典型数据:多轮对话等常见 workload 的 cache 命中率通常在 90% 以上,此时 prefill 工作量通常低于 prompt token 的 10%,流程大多可免于 disaggregation。除此之外正文未给出其他量化数字。

## 现状与落地
已关闭(2026-05-31)。正文为详细设计提案(2025-09-12 更新补充了抢占用例),phase 2 优化计划在核心阈值机制合入后的后续 PR 实现;是否实际合入从正文无法看出,落地情况需查关联 PR。

## 关联
- [llm-d-inference-scheduler issue #382](https://github.com/llm-d/llm-d-inference-scheduler/issues/382)(相关 llm-d 工单与文档)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
