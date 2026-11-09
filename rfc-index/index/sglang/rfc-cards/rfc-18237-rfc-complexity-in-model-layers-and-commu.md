# RFC #18237: [RFC] Complexity in model layers and communication kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18237](https://github.com/sgl-project/sglang/issues/18237) |
| 作者 | nvjullin |
| 创建日期 | 2026-02-04 |
| 状态 | ⚪ closed (2026-05-13) |
| labels | RFC, Refactor, high priority, nvidia |
| 热度 | 👍 6 · 💬 4 |
| 分类 | Attention / Kernel |

## 一句话结论
以 DeepSeek-R1 为例指出层间通信 kernel 决策分散且暗耦合、难以保证一致性,提议由 Model 层统一计算通信算子并注入各层。

## 提案动机
`DeepseekV2DecoderLayer` 的 API 假装独立于其他层,实际在 `__init__` 里偷看相邻层是否 sparse(且只是尽力猜测);`LayerCommunicator` 与 `LayerScatterModes` 各自维护复杂决策树、互为输入,而 `DeepseekV2MoE` 又直接读全局配置而非查询 `LayerScatterModes`。通信 kernel 之间必须一致(`reduce-scatter -> MLP -> all-gather` 与 `all-reduce -> MLP` 各自正确,`reduce-scatter -> MLP -> all-reduce -> all-gather` 则不正确);当前架构「到处做局部决策、无一致性强制」,极难推理通信 kernel 的正确性。

## 提出的方案
- 通信决策改在 `DeepseekV2Model` 统一完成:`compute_selected_ops()` 基于 model/server 配置逐层 select 出 `selected_ops` 并注入每个 `DeepseekV2DecoderLayer`;层完全依赖注入结果,不再自行决策
- `LayerTransform`(如 `DenseLayerTransform`)按 `ShardingMode`(Full / SplitToken / Partial)状态机式推进,select_attn/select_mlp 产出的算子参数与通信强制一致
- 遍历整个 forward,端到端一致性可校验(如 #18232 中对 `SplitToken` 的 hidden_states 做 allreduce 这类错误应容易发现)
- 评估 `torch.DTensor` 并给出四点不适用:无法按调用点控制通信 kernel;需改写全部 torch custom op;不允许融合通信 kernel(如 flashinfer all-reduce fusion);所有通信对其他 kernel 同步、无通信/计算 overlap——第 2 点工程量大,第 3、4 点 likely insurmountable,结论 DTensor 不可行

## 优化点与收益
正文未给出量化数字;目标是一致性/正确性保障与可维护性:把分散的局部决策集中为有限状态机式的端到端一致变换,使通信 kernel 正确性易于检验。

## 现状与落地
已关闭(2026-05-13)。正文为设计提议与 DTensor 可行性分析;落地情况需查关联 PR。

## 关联
- #18232(背景 issue)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
