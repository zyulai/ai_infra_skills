# RFC #49488: [RFC] Persistent, in-place drafting attention metadata for full-CUDA-graph MTP/spec-decode

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49488](https://github.com/vllm-project/vllm/issues/49488) |
| 作者 | chanh |
| 创建日期 | 2026-07-22 |
| 状态 | ⚪ closed(关闭于 2026-07-22) |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
把 MTP/spec-decode 草稿循环的 attention metadata 改为预分配+GPU 就地更新,使 draft 循环可全 CUDA graph 化,消除 host 重建元数据造成的 GPU 空等。

## 提案动机
在 H100 上 profiling Step-3.7-Flash(Step3p5 MTP drafter,`num_speculative_tokens=2`)显示 MTP draft 阶段 GPU 在等 host:`Self CPU time ≈ Self CUDA time`,约 25% 的 decode 时间线是 GPU 空等 kernel launch(nsys `GAP_PERCENTAGE`/`LONG_GPU_WAITING_FOR_KERNEL_LAUNCH`);torch profile 归因于 draft loop 附近大量微小的 host 驱动 ops(`aten::copy_` ×12.7k、分解的 RMSNorm、MoE dispatch)。根因:draft forward 已 CUDA-graph 化,但多步 draft 循环(`llm_base_proposer.py` ~L657–712、`step3p5.py` L385–440)里每个 speculative step 都在 host 上重建 attention metadata(`build_per_group_and_layer_attn_metadata → metadata_builder.build_for_drafting(...)`),并逐步重算 position/slot-mapping,在图化 forward 之间串行化 CPU↔GPU、饿住 GPU。现有 fast path `constant_draft_positions=True` 不适用:MTP 是顺序推进(positions+1、每步重算 slot_mappings),该重建目前是正确性所必需。

## 提出的方案
复用 main-decode 已有的「持久缓冲就地更新」模式,让整个 draft loop 在有动态性的情况下也可 full-cudagraph 捕获:
1. 在最大 padded shape 上一次性预分配 drafting attention metadata 对象及其 tensors(persistent input/hidden/position/slot_mapping 缓冲已存在)。
2. 每个 draft step 在 GPU 上就地更新 metadata tensors(`positions += 1`、用现有 kernel 重算 `slot_mapping`、调整 `seq_lens`),不再在 host 构造新 metadata 对象。
3. 用 pad-to-max + `PADDING_SLOT_ID` masking 吸收 dynamism(被拒 token、可变 seq_lens)。
4. 将整个 draft loop 纳入 full cudagraph capture,使图化 forward 背靠背发射、中间无 host 停顿。
新增 backend hook(与 `build_for_drafting` 并列):`update_for_drafting_inplace(self, md, common_attn_metadata, draft_index) -> None`,先实现 FlashAttention-3(Step-3.7 所用 backend),再扩展其他 backend。

## 优化点与收益
目标是把 per-step host metadata 重建与拷贝移出关键路径,让 GPU 在 MTP 期间不再空转等待。量化的问题规模:~25% decode 时间线 GPU 空等、`aten::copy_` ×12.7k;正文未给出实现后的加速数字。正确性门槛:spec-decode 失败是静默的,验证须确认 mean acceptance length 不变(该配置 ~2.28)且输出与现路径 parity。另注:H100-NVL 上 ~32% 的 TP all-reduce 成本属拓扑问题(bridged-pair 上正确禁用了 custom all-reduce,NVSwitch H100 上消失),不在本 RFC 范围。

## 现状与落地
已关闭(2026-07-22,与创建同日)。正文称「Draft PR: to follow / see linked PR」,未给出结论;是否落地需查关联 PR。

## 关联
正文未列出 RFC/PR 链接(涉及代码路径 `llm_base_proposer.py`、`step3p5.py`、`gemma4.py` 与 `build_for_drafting` hook)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
