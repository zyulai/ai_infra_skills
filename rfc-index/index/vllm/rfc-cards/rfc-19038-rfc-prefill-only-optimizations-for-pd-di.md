# RFC #19038: [RFC]: Prefill-only optimizations for PD disaggregation in vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19038](https://github.com/vllm-project/vllm/issues/19038) |
| 作者 | KuntaiDu |
| 创建日期 | 2025-06-02 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 27 · 💬 12 |
| 分类 | Disaggregated Serving |

## 一句话结论
面向 PD 分离新增 prefill-only 优化:KV 存 CPU 换来约 7x 上下文长度,并利用确定性完成时间改进 prefill 调度。

## 提案动机
vLLM 正走向完全 PD 分离,这使专门针对 prefill 工作负载的优化成为可能:prefill 节点没有解码步骤,生成的 KV cache 可直接存 CPU 而非 GPU,从而获得长得多的上下文(约 7x);同时每个请求的完成时间(job completion time)完全确定,可将其作为调度器输入做更好的调度。更多细节见论文 arXiv:2505.07203。

## 提出的方案
- **用户 API**:新增 `vllm prefill`,只接受 prefill 请求(`max_tokens=1`)并执行 prefill-only 优化。
- **更长上下文(hybrid prefilling)**:把生成的 KV cache 直接存 CPU 并控制长上下文推理峰值内存;不用传统 chunked prefill(它要求保留先前的 KV),而是 attention 层正常 prefill、FFN 模块逐 chunk prefill,可经 `torch.compile` 透明实现。
- **内存分配**:给出两个开放设计——直接绕过 GPU cache、纯用 CPU 作 KV 主缓存层;或保留 GPU cache、仅当请求 KV 放不进 GPU 时才用 CPU。两者均合理,公开征求意见。
- **更好调度**:prefill-only 请求完成时间可估为「输入 token 数 − cache 命中 token 数」;新增 `get_num_computed_blocks` 只返回命中 token 数而非块列表(可用二分查找降本);新调度器按此估计调度,可为 greedy shortest-remaining-job-first,也可为最大化 SLO 满足率。

## 优化点与收益
正文照抄的数字:
- Context length improvement: **~7x on A100 40 GB**,模型 RedHatAI/DeepSeek-R1-DistillQwen-32B-FP8-dynamic(hybrid prefilling 的效果,正文附有效果图)。
- 其余(调度、内存分配)正文未给出量化数字。

## 现状与落地
仍在讨论/征求意见:内存分配的两个设计公开征求反馈,CC 了 @simon-mo、@WoosukKwon、@youkaichao 等维护者;具体落地需查后续关联 PR。

## 关联
- [arXiv:2505.07203](https://arxiv.org/abs/2505.07203)(详细论文)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
