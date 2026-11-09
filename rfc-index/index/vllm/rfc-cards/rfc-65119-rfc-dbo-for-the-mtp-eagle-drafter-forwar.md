# RFC #65119: [RFC]: DBO for the MTP/EAGLE drafter forward in Model Runner V2

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65119](https://github.com/vllm-project/vllm/issues/65119) |
| 作者 | algebra-MCX |
| 创建日期 | 2026-11-02 |
| 状态 | 🟢 open |
| labels | speculative-decoding |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
提议在 Model Runner V2 中对 MTP/EAGLE drafter forward 做 DBO microbatch,让 draft MoE 的 dispatch/combine 与 peer microbatch 重叠。

## 提案动机
DeepSeek 类部署通常 DP+EP + MTP。#61464 合并后 `--enable-dbo` 与投机解码可在 MRV2 结合,但只 microbatch target forward,drafter 仍在整批上运行;DeepSeek MTP 的 draft layer 本身就是 MoE 层,其 dispatch/combine 每个 draft 步都完全暴露、无 peer microbatch 可重叠。此外 ubatching 下 `can_reuse_target_attn_metadata = False`,drafter 每步重建 full-batch attention metadata。作者检索(2026-11-02)确认无在途 PR/issue 覆盖 drafter microbatching,并主动请缨负责设计、实现与验证。

## 提出的方案
第一片范围:CUDA、DP+EP(DeepEP LL 或 NIXL EP)、DeepSeek MTP 单步、uniform decode、eager;drafter FULL CUDA graphs 留待后续。
1. 复用 target 的 `UBatchSlices`(拒绝后的 request 边界与 token 范围),不重算 split;microbatch 数与 target step 一致,无需新 DP collective。
2. draft forward 走 `UBatchRunner`:按 microbatch 切 input ids/positions/hidden states,在现有 ubatch 线程/流运行,采样前合并输出;采样与 draft-token bookkeeping 保持 full-batch 单一所有权。
3. per-microbatch draft attention metadata(顺带消除上述 full-batch 重建)。
4. 回退:target step 未 microbatch、或后续 draft 步无法跨 DP 一致拆分时,整批跑 drafter(同现状)。
范围外:multi-step tree drafting、EAGLE3 auxiliary hidden-state 布局、drafter CUDA graphs。

## 优化点与收益
正文未给出量化数字——明言「尚无实现、GPU run 或性能结果」。完成标准包括:DBO/spec decode 单开或全关时输出不变;2×H200 DP2/EP2 与 8-GPU DP/EP 上开关 drafter microbatching 的 draft token 与 acceptance rate 一致;trace 证明 draft MoE dispatch/combine 与 peer microbatch 重叠;`vllm bench serve` 吞吐/TPOT 与 drafter 步时占比对比。

## 现状与落地
open,反馈期两周,尚无实现;向维护者征询:drafter 是否可复用 target 的 split、DBO+MTP 同开时 drafter microbatching 应默认开还是先 opt-in。

## 关联
PR #61464(其余为仓库内代码位置链接,非 RFC/PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
