# PR #46558: [Bugfix][Attention] Enable FlashInfer mm-prefix attention

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46558](https://github.com/vllm-project/vllm/pull/46558) |
| 作者 | lesj0610 |
| 合并日期 | 2026-11-11 |
| labels | bug, documentation, ready, v1, multi-modality, nvidia |
| 热度 | 👍 3 · 💬 36 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
FlashInfer 后端此前不承接多模态 PrefixLM(mm-prefix)批;本 PR 接入其双向区间包装器,按区间计算注意力,免建 O(qo_len*kv_len) 稠密掩码以省显存。

## 问题现象
正文未描述具体故障现象或报错日志。按正文「Purpose」,此前 vLLM 中 multimodal PrefixLM(mm-prefix)批不能留在原生 FlashInfer 后端:#47118 只对 Gemma 4 多模态模型的纯文本请求启用 FlashInfer,而带图像 mm-prefix 区间的批需要回退到其他路径(如 SM100 上的 `TRITON_FLASHINFER` 复合后端)。影响面为使用多模态 PrefixLM 模型(如 Gemma 系列)且带图像前缀区间的 prefill 批。

## 根因
正文未以「根因」措辞披露,但 Purpose 段说明了缺口所在:vLLM 缺少对 FlashInfer mm-prefix batch-prefill 包装器的适配,该包装器的注意力变体在每个 KV tile 上求值 `(causal AND sliding_window) OR mm_prefix`,按请求元数据构建的每 query `[start, end]` 区间行工作,无需物化 `O(qo_len * kv_len)` 的稠密掩码;prefill 仍遍历完整矩形,收益在内存而非 FLOPs。该包装器来自 FlashInfer #5189,自 0.7.1 起可用,main 的 pin 已包含。

## 修复方案
改 `vllm/v1/attention/backends/flashinfer.py`,并扩展选择器与 E2E 测试:

- 新增谓词 `_native_mm_prefix`,统一决定后端选择与区间缓冲分配:`is_mm_prefix_lm` 且非 DCP、无 `logits_soft_cap`、NVFP4 缓存需包装器支持 `v_sf_swizzled`(来自 FlashInfer #7012,0.7.3 起)、SM100 上仅显式选 `FLASHINFER` 时启用。
- `FlashInferMetadataBuilder` 按scheduled tokens 分配 `int32 (N,2)` 区间缓冲(16384 tokens 约 128 KiB),`build()` 调 `fill_mm_prefix_query_ranges` 填充,并为 mm-prefix 批强制 native paged prefill 与 CPU seq_lens;`forward()` 在 mm 包装器上执行,FP8 query 以 q_scale 无损还原模型 dtype,FP8 输出调用后量化。
- 关键摘录:

```python
return (
    model_config.is_mm_prefix_lm
    # The DCP prefill's context/new-token split breaks the ranges.
    and vllm_config.parallel_config.decode_context_parallel_size == 1
    # The wrapper has no logits soft cap (e.g. Gemma 2 language models).
    and not getattr(model_config.hf_text_config, "attn_logit_softcapping", None)
    ...
)
```

文本批、decode 与 forward 时被清空区间的层(如 Gemma 4 全注意力组)仍走原 causal 路径;选择阶段拒绝 `fp8_e5m2`、`nvfp4_4over6`、sinks 与 DCP。

## 影响与回归风险
- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:按 diff,改动集中于 FlashInfer 后端——`supports_mm_prefix`/`supports_combination`/大页 kernel block 判定(与 `_native_mm_prefix` 挂钩)、`FlashInferMetadataBuilder.build` 的 prefill 路由(`treat_short_extends_as_decodes`、`needs_seq_lens_cpu`、`needs_native_paged_prefill` 条件变化)、`forward()` prefill 分支与 `fused_output_quant_supported`。SM90 自动优先级(`TRITON_FLASH_ATTN` 复合)保持不变;文本-only、decode 路径不受影响,但 mm-prefix 批的 decode/prefill 归类与 FP8 输出融合路径有行为变化,需要回归验证。

## 关联
- [FlashInfer #5189](https://github.com/flashinfer-ai/flashinfer/pull/5189)
- [FlashInfer #7012](https://github.com/flashinfer-ai/flashinfer/pull/7012)
- [#47118](https://github.com/vllm-project/vllm/pull/47118)
- [#65930](https://github.com/vllm-project/vllm/pull/65930)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
