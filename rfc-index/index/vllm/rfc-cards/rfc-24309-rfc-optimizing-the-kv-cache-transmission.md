# RFC #24309: [RFC]: Optimizing the KV cache transmission mechanism in PD disaggregation to reduce its impact on TTFT/TPOT.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24309](https://github.com/vllm-project/vllm/issues/24309) |
| 作者 | CalvinXKY |
| 创建日期 | 2025-09-05 |
| 状态 | ⚪ closed(关闭于 2026-01-05) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议 PD 分离下把首 token 生成挪到 Decode 侧并与 KV cache 传输重叠,降低传输对 TTFT/TPOT 的影响。

## 提案动机
PD 分离场景的常见做法是等 Prefill 计算全部完成后才开始传输 KV cache。首 token 由 P 生成没有问题,但这种做法把 KV cache 的传输时间计入其中,影响第二个 token 的生成时间,从而影响 TTFT/TPOT。

## 提出的方案
核心是把首 token 的生成移到 Decode 阶段(正文附流程图),步骤如下:
- 1.1 KV cache 计算完成后即可发起逐层(layer-wise)传输,也可以等全部计算完成后再传。
- 1.2 P 的最后一层计算完成后,将 hidden state 传给 D 实例。
- 2 D 实例执行首 token 的计算。
- 3 KV cache 传输完成后,decode 阶段继续生成后续 token。
- 步骤 1.1 与 1.2/2 可以并发执行。

收益:P 节点在 transformer blocks 计算完成后即可省去其后所有层的部分,首 token 的后续计算与 KV cache 传输得以重叠进行,不再串行等待整段传输结束。

## 优化点与收益
正文未给出量化数字。定性收益:通过「首 token 计算与 KV cache 传输重叠」消除传输时间对第二个 token 生成时间的串行占用,从而降低 PD 分离中 KV 传输对 TTFT/TPOT 的影响。

## 现状与落地
已关闭(关闭于 2026-01-05),labels 含 stale。正文较简短,Feedback Period 一栏为 No response,未列出实现 PR,也未写明方案是否被实现或替代;落地情况需查关联 PR。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
