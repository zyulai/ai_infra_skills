# RFC #65122: [RFC]: Zero-copy DeepEP low-latency combine, safe under DBO

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65122](https://github.com/vllm-project/vllm/issues/65122) |
| 作者 | algebra-MCX |
| 创建日期 | 2026-11-02 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |

## 一句话结论
为 DeepEP 低延迟 combine 启用零拷贝:专家输出直接写入下一个 combine buffer,省去每个 MoE 层每个 decode step 的一次拷贝。

## 提案动机
DeepEP LL combine 目前以 `zero_copy=False` 运行,DeepEP 先把 expert 输出拷贝进自己的通信缓冲。这个拷贝发生在每个 decode step 的每个 MoE 层上——而 LL 正是为这条延迟关键路径设计的;零拷贝模式下专家把输出直接写进下一个 combine buffer。当前 vLLM 代码(commit `5689c279`)中,`deepep_ll._finalize` 处留有 `# TODO (varun) : Enable zero copy mode`,随后调用 `low_latency_combine(..., zero_copy=False)`;而 `allocate_fused_expert_output` 钩子(让 prepare/finalize 提供 fused-expert 输出缓冲)已存在,`naive_dp_ep` 与 `mori_epv2` 已在用,DeepEP LL 尚未使用。作者检索(2026-11-02)未发现实现该功能的开放 PR 或 issue。

## 提出的方案
第一切片范围:CUDA、DeepEP LL、batched experts 且输出布局与 DeepEP combine buffer 匹配(`[num_local_experts, num_ranks * max_tokens_per_rank, hidden]`,BF16),DBO 关与开两种情形:
1. **提供输出缓冲**:在 DeepEP LL prepare/finalize 中实现 `allocate_fused_expert_output`,布局与 dtype 匹配时返回 `get_next_low_latency_combine_buffer(handle)` 的视图,否则返回 `None` 走现有拷贝路径;
2. **`zero_copy=True` 仅在专家确实写入该缓冲时启用**;
3. **DBO 缓冲所有权**:DeepEP 每次 call 交替使用 LL buffer,DBO 下对端 microbatch 可能在本 microbatch 的 buffer 分配与 combine 之间发起 LL dispatch、改变"下一个" buffer 是谁。设计必须为每个 microbatch 固定所用 buffer(或在 DBO 下回退拷贝路径),确保 microbatch 绝不写入对端正在使用的 buffer——这是主要开放问题,决定 DBO 是否纳入第一切片。
- 范围外:对 DeepEP 本身的改动、DeepEP v2 与 HT backend。
- 完成标准(正文清单):布局不匹配时默认行为不变;unit/kernel 测试中零拷贝与拷贝路径 combine 结果一致、交错 microbatch 调用下缓冲所有权正确;2×H200 DP2/EP2 + DeepSeek-V2-Lite(BF16 与 FP8)、FULL CUDA graphs 下零拷贝开/关、DBO 开/关的输出与 logprobs 完全一致;测量每层 combine 时间与 decode TPOT(零拷贝开/关)。

## 优化点与收益
正文未给出量化数字(作者明言尚无实现、GPU 运行或性能结果)。定性收益:消除 LL 延迟关键路径上每个 MoE 层、每个 decode step 的一次 expert 输出拷贝;具体收益以完成标准中规划的每层 combine 时间与 decode TPOT 对比为准。

## 现状与落地
仍在等待维护者反馈(open,反馈期两周)。作者表示将主导实现与验证(subject to maintainer feedback);尚无实现、GPU 运行或性能结果;调查与起草使用了 AI 辅助。正文向维护者提出两个问题:(1) 原 TODO 是否仍想做,还是 zero-copy 因已知原因被有意禁用;(2) DBO 下固定每个 microbatch 的 combine buffer 是否可接受,还是先只支持 DBO-off。

## 关联
- [deepep_ll.py `_finalize`](https://github.com/vllm-project/vllm/blob/5689c2790c911a55c75825aac5166305df0b591d/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py#L393-L404)(zero_copy=False 与 TODO 所在)
- [modular_kernel.py `allocate_fused_expert_output`](https://github.com/vllm-project/vllm/blob/5689c2790c911a55c75825aac5166305df0b591d/vllm/model_executor/layers/fused_moe/modular_kernel.py#L203)(待复用的既有钩子)
- 正文未列出其他 RFC/issue/PR 链接(作者检索未发现相关实现 PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
