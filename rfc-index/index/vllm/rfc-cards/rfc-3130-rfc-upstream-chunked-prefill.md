# RFC #3130: [RFC] Upstream Chunked Prefill

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3130](https://github.com/vllm-project/vllm/issues/3130) |
| 作者 | rkooo567 |
| 创建日期 | 2024-03-01 |
| 状态 | ⚪ closed (2024-04-26) |
| labels | (none) |
| 热度 | 👍 15 · 💬 18 |
| 分类 | Overlap / Scheduler |

## 一句话结论
把 Anyscale 内部已在生产启用的 chunked prefill 上游化到 OSS vLLM:prefill 切块与 decode 混批,大幅改善高 QPS 延迟。

## 提案动机
chunked prefill 把 prefill 请求切成多个 chunk 与 decoding 请求混合成批:prefill 是计算受限(compute-bound)、decoding 是内存受限(memory-bound),两者重叠执行可大幅提升系统效率(参见所链接的两篇论文)。Anyscale 已在内部 vLLM 上实现该功能并在生产端点启用,本 RFC 计划将其上游化到 OSS vLLM 仓库。

## 提出的方案
- 内核:内部实现使用 flash attn + paged attention,上游决定先用现有 context attention 再迁移到 flash infer(内核选择经过调整,最初考虑先上游 flash attn 集成);
- 查询布局:chunked prefill 混合 prefill 与 decoding 后,带 batch 维度的 2D query 低效,先做 2D→1D query 重构;
- 调度器与批布局:`Sequence` 记录 prompt token 已 prefill 的数量;启用 chunked prefill 时调度器优先调度 decoding 请求、再挑 prompting 请求,过长的 prompt 会被切块;worker 为 decoding 与 prompting 请求统一创建元数据(block table、context 长度、decoding/prefilling token 数等),输入按 prompt 段 + generation 段平铺,并记录 `cum_prompt_context_lens` 等字段;
- 采样结果:请求处理并采样后,属于 chunked prefill 请求的那部分结果将被忽略;
- CUDA Graph:不支持 chunked prefill(其收益只在 token 数较小时显现),"Enable cuda-graph on prefill path" 一项标注 Won't do;
- 未来扩展(不属于本 RFC 范围):兼容 sliding window attention、复用同一机制做 prefix caching、换用更好的内核(理想为 flash infer)。

## 优化点与收益
基准条件:Llama 13B x 2 A100,不同 QPS(结果来自 Anyscale fork 的 vLLM,详见正文链接的 benchmark 评论)。结论为:QPS 高时 chunked prefill 大幅改善延迟,QPS 低时性能相当;OSS vLLM 的详细分参数基准仍在进行(ETA 当周周末)。具体数值在正文引用的图与评论中,pack 文本未给出量化数字。

## 现状与落地
已关闭(2024-04-26)。正文 Progress 部分列出一连串配套工作:2D→1D query 重构、Sequence group metadata API 更新(#3538)、Scheduler 重构(#3550)、chunked prefill 调度器(#3853)、chunked prefill attention 更新(#3884)、logprob 修复(#4309);功能落地情况需查这些关联 PR。

## 关联
- https://arxiv.org/pdf/2401.08671.pdf (相关论文)
- https://arxiv.org/pdf/2308.16369.pdf (相关论文)
- https://github.com/vllm-project/vllm/issues/3130#issuecomment-2011281519 (基准结果详情)
- https://github.com/vllm-project/vllm/pull/3853 (chunked prefill scheduler)
- https://github.com/vllm-project/vllm/pull/3884 (chunked prefill attention update)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
