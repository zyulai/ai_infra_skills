# PR #52188: [Spec decode] Support Kimi-K3 DCP with DSpark

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-08-17 |
| 分类 | Attention / Kernel (tier: major) |
| Label | speculative-decoding, ready, nvidia, mrv2, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/52188 |

## 一句话结论

解除 Kimi-K3 + DSpark 与 decode context parallel(DCP)的组合禁令:目标/草稿注意力分别支持 FlashInferMLA 与 Tokenspeed 后端,TP8+DCP8 下 gsm8k 0.96+。

## 动机

此前 `SpeculativeConfig` 对 K3DSparkModel 的 MLA DSpark 组合直接抛 ValueError("MLA DSpark does not currently support decode context parallelism; set decode_context_parallel_size=1"),Kimi-K3 在 DSpark speculative decoding 下无法用 DCP 切分 decode KV,长上下文高并发的显存/吞吐受限于单 DCP 组。需要让目标 causal 注意力与 draft 非 causal 注意力在 DCP KV 布局下都可用。

## 优化点

机制分四层:(1) 放开配置并改为按后端能力校验——删除 spec config 里的硬性拒绝,在 `MLACommonMetadataBuilder` 上新增 `supports_non_causal_multi_token_dcp` 能力位与 `_validate_dspark_dcp_support`:DSpark + DCP 时,非 causal draft 路径要求 `supports_non_causal_multi_token_dcp`,causal 多 token 路径要求 `supports_dcp_with_varlen`,不满足则给出指向后端选择或 `decode_context_parallel_size=1` 的报错;`AttentionBackend.validate_configuration` 增加 `use_dcp` 判据,selector 传入 DCP>1。(2) FlashInferMLA 支持 causal DCP 多 token decode——`forward_mqa` 在 `dcp_world_size > 1 且 query_len > 1` 时把整个 DSpark 块摊平为单 token decode 行,并用 `_prepare_flattened_decode_metadata` 从 `dcp_tot_seq_lens` 按 `query_len` 反推每个 query 的可见前缀,再按 `dcp_world_size * cp_kv_cache_interleave` 的 span 与 rank/interleave 取模算出 rank-local `seq_lens`;摊平结果缓存在类型化的 `FlashInferMLADecodeMetadata`(flattened_block_table/seq_lens/query_len)里供同组所有层复用。(3) Tokenspeed 作为 draft 非 causal 后端——其 builder 声明 `supports_non_causal_multi_token_dcp = True`(内核接受显式 causal mask,块可保持融合不摊平)。(4) DFlash/dspark 输入准备 DCP 化——`_prepare_dflash_inputs_kernel` 增加 cp_rank/cp_size/cp_interleave,block 定位改按 `ctx_pos // (block_size * CP_SIZE)`,slot 经新 Triton 内核 `cp_local_slot` 折算成本 rank 本地 slot(非本 rank 的位置写 PAD_SLOT_ID),`dcp_local_seq_lens` 贯通 draft attention metadata 与 CUDA graph capture 路径;Kimi-K3 MLA 里 "DCP 不支持 RoPE" 的 assert 一并移除。

## 关键改动

```text
vllm/config/speculative.py: 删除 K3DSparkModel + DCP 的硬性拒绝
vllm/model_executor/layers/attention/mla_attention.py: supports_non_causal_multi_token_dcp 能力位 + _validate_dspark_dcp_support
vllm/v1/attention/backends/mla/flashinfer_mla.py: causal DCP 摊平 + rank-local seq_lens 计算,metadata 类型化缓存
vllm/v1/worker/gpu/cp_utils.py + spec_decode/dflash/*: cp_local_slot Triton 内核与 prepare inputs 的 DCP 折算
vllm/v1/attention/backends/mla/tokenspeed_mla.py + backend.py + selector.py: DCP 能力声明与选择校验
```

关键摘录(causal DCP 的 per-query rank-local len):

```python
per_query_global_lens = torch.clamp(
    (global_seq_lens.unsqueeze(1) - offsets).reshape(-1), min=0)
interleave = self.cp_kv_cache_interleave_size
dcp_span = self.dcp_world_size * interleave
remainder = torch.clamp(
    per_query_global_lens % dcp_span - self.dcp_rank * interleave,
    min=0, max=interleave)
seq_lens = per_query_global_lens // dcp_span * interleave + remainder
```

## 性能声明

无吞吐/延迟数字;PR 正文给出 Kimi K3 gsm8k(5-shot)精度对照,服务配置 TP8 + `-dcp 8` + fp8 KV cache + DSpark `num_speculative_tokens=4` + fastsafetensors:默认后端 0.9606 ± 0.0054(flexible/strict 同值);FlashInferMLA target + Tokenspeed draft 0.9613 ± 0.0053;Tokenspeed target + Tokenspeed draft 0.9621 ± 0.0053。硬件型号未注明(标签含 nvidia)。

## 验证与回归风险

测试:删除 `test_dspark_mla_rejects_decode_context_parallelism`(行为反转);新增 `test_flashinfer_mla_dcp_multi_token_decode_uses_per_query_bounds`(校验摊平后 per-query seq_lens [4,4,5,5,6,6] 与 repeat 的 block_tables、query 形状);`test_prepare_dflash_inputs_excludes_rejected_context_suffix_with_dcp`(cp_rank=1/cp_size=2/cp_interleave=2 下的本地 slot 与 PAD);`test_tokenspeed_mla_noncausal_capability` 扩展 DCP 能力断言;`test_flashinfer_mla_forward_uses_gathered_head_count` 增加 dcp_world_size 字段。回归风险:`FlashInferMLAMetadata` 从 `MLACommonMetadata` 换成类型化子类,所有经该后端的 MLA 模型(非仅 Kimi-K3)受 metadata 形状影响;causal DCP 摊平的 rank-local lens 数学依赖 cp interleave 语义;`validate_configuration` 新增的 "non-causal MLA attention with DCP not supported" 会使此前静默可用的组合显式失败(报错而非崩溃);`_prepare_flattened_decode_metadata` 的缓存假设同层组 query_len 一致。

## 关联

同区域:Kimi-K3 MLA(vllm/models/kimi_k3/nvidia/mla.py)、DSpark/DFLash spec decode 路径、DCP/PCP 并行(PR #53867 同批次涉及 PCP decode-only FULL CUDA graphs)。上游依赖 FlashInfer trtllm MLA decode。本卡所在批次素材中未列出其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
