# PR #21078: [Kernel] Flashinfer MLA (trtllm-gen) decode kernel integration

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hjjq |
| 合并 | 2025-09-10 |
| 分类 | Attention / Kernel |
| Label | ready, ci/build, v1 |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/21078 |

## 一句话结论

为 DeepSeek MLA decode 接入 FlashInfer 的 trtllm-gen kernel(新 `FLASHINFER_MLA` 后端):高并发下 kernel 比 CUTLASS MLA 快 25%。

## 动机

Blackwell(cc ≥10.0)上 vLLM V1 已有 CUTLASS_MLA 作为 MLA decode 后端,但缺 FlashInfer 封装的 trtllm-gen MLA decode kernel 选项。高并发 decode 下 MLA kernel 是重要开销,需要一个更快的 kernel 供用户显式选择;E2E 中 MLA 并非瓶颈,故收益主要体现在 kernel 层。

## 优化点

新增 `FlashInferMLABackend`/`FlashInferMLAImpl`(继承 `MLACommonBackend`/`MLACommonImpl`),`_forward_decode` 直接调 `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`:query 与 kv_cache 各 unsqueeze 出 `q_len_per_request` 维以支持 MTP,workspace 用模块级共享的 128 MiB buffer。路由上,`_Backend` 枚举新增 `FLASHINFER_MLA`,进 `arg_utils` 支持白名单;`platforms/cuda.py` 在用户显式选择且 cc≥10、V1 引擎时启用,并 `set_kv_cache_layout("HND")` 以匹配 trtllm API 的 KV 布局。`MLACommonMetadata` 增加 `max_seq_len` 字段传给 kernel。不支持项在构造时显式 raise:alibi_slopes、sliding_window、logits_soft_cap、FP8 KV cache、encoder/cross-attention。

## 关键改动

```text
vllm/v1/attention/backends/mla/flashinfer_mla.py: 新 backend 与 decode impl
vllm/platforms/cuda.py + vllm/platforms/interface.py: FLASHINFER_MLA 枚举与 cc>=10 路由
vllm/v1/attention/backends/mla/common.py: metadata 增加 max_seq_len
tests/kernels/attention/test_flashinfer_mla_decode.py: DeepSeek R1 形状数值测试
vllm/engine/arg_utils.py: 支持白名单加入 FLASHINFER_MLA
```

```python
o = trtllm_batch_decode_with_kv_cache_mla(
    query=q,
    kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
    workspace_buffer=self._workspace_buffer,
    qk_nope_head_dim=self.qk_nope_head_dim,
    kv_lora_rank=self.kv_lora_rank,
    qk_rope_head_dim=self.qk_rope_head_dim,
    block_tables=attn_metadata.decode.block_table,
    seq_lens=attn_metadata.decode.seq_lens,
    max_seq_len=attn_metadata.max_seq_len,
    bmm1_scale=self.scale,
)
```

## 性能声明

- E2E:DeepSeek-R1,TP8,`--block-size 32 --max-num-seqs 512`,random 1024 in/2048 out,1024 prompts、并发 512:trtllm-gen MLA output tok/s 4827.98(total 7235.94)vs CUTLASS_MLA 4605.44(6902.40),约 +4.8%;TTFT/TPOT/ITL 亦略优。PR 注明 E2E 不被 MLA bottleneck,故 E2E 提升不显著。
- Kernel microbenchmark:bs=512 时 trtllm-gen 0.501 ms vs cutlass 0.626 ms(高并发下快 25%);bs=1 时 0.023 vs 0.025 ms。
- 精度(lm-eval,DeepSeek-R1):gsm8k 0.97 vs cutlass 0.95(`--limit 100`),gpqa 0.47 vs 0.47,mmlu 0.85 vs 0.85。

## 验证与回归风险

新增 `test_flashinfer_mla_decode.py`:bs 1/2/4/16、block_size 32/64,DeepSeek R1 形状(128 heads、kv_lora_rank 512、nope 128/rope 64),与 SDPA reference 对比 atol/rtol 1e-2,cc<10 时整文件 skip;cutlass MLA 测试移入 `tests/kernels/attention/` 并双双纳入 Buildkite pipeline。风险:仅 cc≥10.0 + V1;选中该后端时全局 KV cache layout 被设为 HND;暂不返回 LSE(等 FlashInfer API);128 MiB workspace 在模块导入时即分配;不支持 FP8 KV cache 与上述 raise 项。

## 关联

索引同区域另有 pr-13747(FlashMLA integration)、pr-20769(SM100 CUTLASS MLA decode 放宽限制);素材未提及本 PR 的前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
