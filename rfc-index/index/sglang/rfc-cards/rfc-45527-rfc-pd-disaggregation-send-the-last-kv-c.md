# RFC #45527: [RFC] PD disaggregation: send the last KV chunk from later PP stages as soon as their forward finishes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#45527](https://github.com/sgl-project/sglang/issues/45527) |
| 作者 | eaglewangao-us |
| 创建日期 | 2026-10-21 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Disaggregated Serving |

## 一句话结论
PP prefill 下让 stage 1..N-1 前向一结束就发最后一块 KV,不等 batch result 绕环,显著降低 TTFT。

## 提案动机
带流水线并行的 PD prefill 服务器上,每个 PP stage 都在处理 batch result 时才发送请求最后一块 KV;result 沿 PP 环每循环迭代走一跳,stage k 比 stage 0 晚 k 个迭代才处理,而 decode 要等所有 stage 报告后才启动。负载高时,最后一级的报告比它持有的 KV 就绪时刻晚约 pp_size - 1 个迭代——每次迭代都是一次完整微批前向,延迟随负载与 PP 深度增长(低负载时环转得快、延迟小)。

## 提出的方案
- Launch 时捕获:微批启动时记录其中跑最后一块 chunk 的请求及 launch event;必须在 launch 时记录,否则下轮调度后 chunked 请求的 extend_end 已指向下一块,会把倒数第二块误判为最后一块。
- Stage k≥1 下一迭代:先 checkpoint_disagg_prefill(各 stage 在循环同一位置执行,radix 树状态跨 stage 一致,后续结果处理时的 checkpoint 变为 no-op),再以 launch event 为 transfer wait event 提前发送最后一块 KV(含 SWA、Mamba、DSA 等每请求状态);复用 _early_send_wait_event / TransferKVChunk.wait_event 现有路径(#45524 已用于非末块),调度线程不阻塞;跳过 set_buf 与 send_aux,KV 写完即报 KVPoll.Success。
- Stage 0 保持现状:在结果处理时写 aux(first token 等元数据)后发末块;decode 在全部 stage 成功后才读 aux,保证读到完整 aux。
- Bootstrapping(optimistic prefill)、health-check 请求及未覆盖组合回退现有路径。
- 先挂 opt-in flag(如 SGLANG_DISAGG_PP_EARLY_LAST_CHUNK),CI 覆盖后默认开启;abort、grammar 失败、sampling-mask 完成仍由结果处理决定,stage 0 不会为 prefill 丢弃的请求报成功。
- 与 release 共识(#36781、#45085)是不同路径:那条线管 prefill 何时返回响应并释放资源,本提案改 decode 何时能启动。

## 优化点与收益
数字来自基于 v0.5.12 的下游分支实现(非 main,只说明效应量级):两 prefill 实例各 PP4×CP4(16 ranks),chunked prefill 4096,--pp-async-batch-depth 1;大 MoE 模型,提示约 80K token(72K 前缀已在 prefill 缓存、8K 新增),1K 输出;闭环并发 64、1280 请求:
- TTFT(稳态,剔除前 64 请求):P50 3.57 s → 2.74 s,P90 4.45 s → 3.66 s(约降 0.8 s),Mean 3.74 s → 2.97 s;重复实验复现;KV 提前约 0.8 s 到达 decode,TPOT 不变。
- 低负载无差异(符合预期)。
- 准确率:passcode-retrieval 12/12、GSM8K 0.960 不变。作者特意加了把 passcode 放进最后一块(深度 0.998)且每请求变更的测试——第一版实现里步骤 1 描述的 bug 只有这个测试能抓到(普通检索测试会因连续请求复用相似文本页而漏检)。

## 现状与落地
仍在讨论(open)。作者明确「先就设计达成一致再写上游 PR」;开放问题:仅 stage 0 写 aux 是否可接受(aux 是否有 stage 特有内容)、后端 decode 侧对「后级已报成功但 stage 0 失败/abort」的失败处理、NIXL 带内完成通知与 Ascend 适配、与 staging buffer、#45337、#45085 的交互、flag 命名与默认值。计划:PR1 Mooncake+Mori(flag 后,含 launch 捕获/单 aux writer 单测、末块检索测试与 PP4 基准)、PR2 NIXL(依赖 #43299)及其他后端,CI 覆盖后默认开启。

## 关联
- #36781(release 共识): https://github.com/sgl-project/sglang/issues/36781
- #45085(release 共识 rework): https://github.com/sgl-project/sglang/issues/45085
- #45524(非末块 early send): https://github.com/sgl-project/sglang/issues/45524
- #43885(PP roadmap): https://github.com/sgl-project/sglang/issues/43885
- #43699(PD roadmap): https://github.com/sgl-project/sglang/issues/43699

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
