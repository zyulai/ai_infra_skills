# RFC #36948: [RFC] [Feature]: PD Disaggregation with MooncakeConnector Roadmap

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36948](https://github.com/vllm-project/vllm/issues/36948) |
| 作者 | dtcccc |
| 创建日期 | 2026-03-13 |
| 状态 | ⚪ closed(关闭于 2026-10-29) |
| labels | feature request, stale |
| 热度 | 👍 12 · 💬 8 |
| 分类 | CI / Infra / Process |

## 一句话结论
为 vLLM 的 MooncakeConnector PD 分离给出功能路线图,规划更高级的并行策略与传输优化。

## 提案动机
在 Mooncake Transfer Engine 成功初步集成进 vLLM(#24718、#31034)之后,本 RFC 提出一份后续功能路线图,以解锁更高级的能力、增强性能并提高灵活性。正文指出:已有的基础工作支持基本的 KV cache offloading,但要充分发挥 Mooncake 的潜力、支撑更复杂的大规模推理场景,需要实现更精细的并行策略与优化。

## 提出的方案
以 checklist 形式给出的核心特性路线图(括注关联 PR/issue,勾选状态照抄正文):
- **已完成**:Mooncake store integration(#38474、#40900);Heterogeneous TP support(#36869);PP support(#44528);kv events & stats(#40414)。
- **未完成**:Layer-wise Transfer;Cross-Layer KV Support(#41093);PCP/DCP Support;hybrid attention。
- **其他改进与修复(均未勾选)**:更好的错误处理;physical block != config block;每层不同 hidden size;HND-NHD transfer。
- Related issue:#33702。

## 优化点与收益
正文未给出量化数字。收益为能力与工程层面:在基本 KV offloading 之上扩展 PD 分离部署的并行维度(异构 TP、PP、PCP/DCP、层间/跨层传输),并改进传输细节(HND-NHD、块大小不一致、每层 hidden size 差异等)以适配更大规模的推理场景。

## 现状与落地
已关闭(2026-10-29)。正文 checklist 显示 Mooncake store integration、Heterogeneous TP、PP support、kv events & stats 四项已通过关联 PR 落地;Layer-wise Transfer、Cross-Layer KV、PCP/DCP、hybrid attention 等仍为未完成状态,关闭原因正文未说明,落地情况需查关联 PR。

## 关联
- #24718、#31034(Mooncake Transfer Engine 初始集成)
- #36869(Heterogeneous TP support)
- #44528(PP support)
- #41093(Cross-Layer KV Support)
- #33702(Related issue)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
