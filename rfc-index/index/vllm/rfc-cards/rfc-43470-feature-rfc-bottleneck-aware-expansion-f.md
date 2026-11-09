# RFC #43470: [Feature]: [RFC] Bottleneck-aware expansion for PD disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43470](https://github.com/vllm-project/vllm/issues/43470) |
| 作者 | zsn6493 |
| 创建日期 | 2026-05-23 |
| 状态 | ⚪ closed(关闭于 2026-09-25) |
| labels | feature request, stale |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Disaggregated Serving |

## 一句话结论
在不改 PD 分离架构的前提下增加瓶颈感知扩容层:运行时识别 prefill/decode/KV 哪侧是主导瓶颈,优先扩容最有效的一侧。

## 提案动机
PD 分离已给出 prefill 与 decode 池的重要隔离,KV transfer、请求迁移、offloading 等工作也在正确方向上推进,但部署仍多为静态供给,容易出现:prefill 饱和而 decode 相对空闲;decode 饱和而 prefill 利用不足;decode 侧 KV 容量在算力尚有余时成为主导瓶颈;远程 KV transfer 积压在两池之间制造隐性停顿。作者认为:PD 解决了混合池的相互干扰,但尚未解决池间的动态不均衡——缺少一个在已分离部署中决定"先扩哪一侧"的策略/运行时层。

## 提出的方案
引入 PD 分离的瓶颈感知扩容层(bottleneck-aware expansion layer),核心是保持现有 PD 架构不变,把运行时压力分为三类并映射到最有效的扩容动作:
1. `Prefill-bound` → 增加预填充容量(prefill 侧算力或内存压力主导);
2. `Decode-bound` → 增加 decode 容量(decode 侧 token 吞吐或调度压力主导);
3. `Decode-KV-bound` → 增加 decode 侧 KV 容量/分层(KV 持有容量主导,即使 decode 算力尚未饱和)。
建议的首步范围:暴露结构化的运行时压力信号;按上述三类对瓶颈分类;支持一个最小的扩容 hook 或 policy target。做到这些就足以让 PD 更自适应,无需一步到位的运行时弹性。
Non-goals:不替换 PD 分离;不做通用运行时 GPU 热插拔;不在活跃 worker 组内做 TP/NCCL 拓扑变更;不在 vLLM 内做全集群 autoscaling。

## 优化点与收益
正文未给出量化数字。收益为部署弹性:在固定拓扑的内部竞争加剧之前,先扩容最有效的一侧、缓解用户可感知的竞争,使 PD 部署对单侧成为主导瓶颈的情形更自适应。

## 现状与落地
已关闭(关闭于 2026-09-25,labels 含 stale)。正文以 4 个待反馈问题收尾:该方向是否符合当前 PD roadmap;三类瓶颈中哪类最适合作为首选目标;首步只做信号暴露还是附带一个最小扩容 hook;如何与进行中的 KV transfer/请求迁移工作交互。正文内未见对这些问题的回答,落地情况需查关联 PR。

## 关联
- #33702(PD disaggregation roadmap)
- #36923(KV transfer push-model 讨论)
- #19854(KV cache offloading RFC)
- #38260(multi-tier offloading RFC)
- #19777(NIXL handshake / autoscaling decode 讨论)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
