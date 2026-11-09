# RFC #47919: [RFC]   guard-gated sparse MLA topology index policy

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47919](https://github.com/vllm-project/vllm/issues/47919) |
| 作者 | teerthsharma |
| 创建日期 | 2026-07-07 |
| 状态 | ⚪ closed(关闭于 2026-11-07) |
| labels | stale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
把 #38725 的拓扑感知 KV 压缩收窄为可证伪的稀疏 MLA 索引策略:默认关闭、带质量 guard、可回退。

## 提案动机
#38725 提出宽泛的拓扑感知 KV cache 压缩框架(生命周期压缩、方向性剪枝、面向 NPU 的矩阵操作、预期的内存/FLOPs 收益)。本 RFC 有意取代该框架,收窄为单一可验证目标:仅测试"用拓扑 witness 替换已有 learned sparse-MLA `topk_indices_buffer` 的有界尾部"能否在同等预算下提升稀疏注意力质量,同时保持 vLLM 现有稀疏 MLA 坐标契约与 fail-closed 行为;不引入新的 KV 压缩栈。

## 提出的方案
- 拓扑对象:`[num_tokens, topk]` 的请求级逻辑 token 索引;保留 learned 前缀,只做有界尾部替换,不做无界的 learned token 驱逐。
- 契约:默认禁用;`torch.int32` 2D 坐标、逻辑请求级偏移(非物理 slot)、保留 `-1` sentinel padding、候选集无重复逻辑 token;dtype/shape/device/预算/坐标/guard 不匹配一律回退 learned 索引。
- 对照(null models):同预算 random、locality、learned-tail 选择器;质量 guard 用相对 L2、cosine、KL、top-1 与 next-token agreement。
- PR 前证伪门槛:契约单测、CPU 参考策略、向量化/内核侧 schedule 路径、`vllm bench` 与稀疏 MLA backend 对比、可观测计数器(保留/插入/回退原因/schedule 耗时)。
- 首个 PR 只加隔离 helper、默认关闭的 config、结果/回退对象与测试,不接运行时服务。

## 优化点与收益
正文明确"screening-only local results,not a vLLM runtime speedup claim"。条件:CPU `distilgpt2`。seq48/budget8:topology index 相对 L2 `0.4263`(优于 random `0.4593`),但 speedup vs dense 仅 `3.142x`,低于 random 的 `3.810x` 与 learned-tail 的 `3.864x`。seq64/budget16:topology schedule 耗时 `104.609 ms`,总时延 `145.37 ms`,speedup `0.931x` —— 原型 schedule 开销抵消加速;该轮 topology 的 mean L2/KL 优于 null 模型,但 guard 仍正确拒绝了全部稀疏策略(topology 另因 speed/schedule overhead 被拒)。

## 现状与落地
已关闭 —— 正文处于提案阶段且明确首个 PR 不接运行时;关闭时正文未给出是否进入实现,落地情况需查关联 PR。

## 关联
- #38725(被本 RFC 收窄取代的宽泛拓扑感知 KV 压缩提案,正文未附链接)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
