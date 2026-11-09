# PR #24197: [flashinfer] [kernel] support for fp8 kv cache for trtllm prefill attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mxz297 |
| 合并 | 2025-09-11 |
| 分类 | Quantization (tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/24197 |

## 一句话结论

让 trtllm prefill attention 支持 fp8 KV cache:prefill 前用 Triton kernel 把涉及的 KV 页反量化成 mock cache,解除"bf16 Q + fp8 KV 不可用"限制。

## 动机

trtllm prefill attention 内核本身不支持 bf16 Q + fp8 KV 的组合,此前 `use_trtllm_attention` 在 `is_prefill and kv_cache_dtype.startswith("fp8")` 时直接返回 False,使 fp8 KV cache 用户在 prefill 阶段被迫走其他后端。作者的观察是:prefill 实际涉及的 token 数相对 KV cache 总容量很小,若只为这些 token 构造一个反量化后的临时 KV,开销相对较小,即可在 prefill 上启用 trtllm 路径。

## 优化点

新增 Triton kernel `_trtllm_prefill_attn_kvfp8_dequant`:按 prefill 的 block table 逐页从 fp8 KV cache 读出 K、V,分别乘 `k_scale`/`v_scale` 反量化到 bf16/fp16,写入"mock KV cache"(只含本 batch prefill 需要的页,页号顺序重编),同时生成顺序编号的 mock block table;`forward` 中当 `q_data_type` 非 fp8 而 KV 为 fp8 时,把 mock cache 传给 `trtllm_batch_context_with_kv_cache`。配套改动:删除 `use_trtllm_attention` 里 prefill+fp8 KV 的禁用分支;新增 env `VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION`,置 1 时 fp8 KV 下不再把 Q 量化到 fp8(`q_data_type` 保持 model dtype),即走"bf16 Q + 本 PR 的 dequant KV"组合。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: 新增 Triton dequant kernel 与 trtllm_prefill_attn_kvfp8_dequant 包装,forward 接入 mock cache
vllm/utils/flashinfer.py: 删除 prefill+fp8 KV 的 trtllm 禁用分支,新增 flashinfer_disable_q_quantization
vllm/envs.py: 新增 VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION 环境变量并纳入 compute_hash
```

mock cache 构造核心:

```python
# mock kv cache contains just the pages needed by this prefill
mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, ...)
mock_block_table = torch.arange(
    start=1, end=batch_size * num_of_page_per_token + 1, ...
).reshape(batch_size, num_of_page_per_token)
```

## 性能声明

PR 正文未给出吞吐/延迟 benchmark 数字,仅以"prefill 涉及 token 远小于 KV cache 总量,dequant 开销相对较小"作定性论证。给出的量化数据是 AIME25 正确性:gpt-oss-120b,temp 1.0,low/medium/high reasoning effort 的 metric 分别为 0.5 / 0.7417 / 0.9。硬件等测试条件 PR 未注明。

## 验证与回归风险

Test Plan 为 AIME25 评测(上述三档结果),diff 中未新增单元测试文件。回归风险:所有走 FlashInfer backend + trtllm attention + fp8 KV cache 的 prefill 路径行为改变——多一次 mock cache 反量化(显存上临时占用 batch×页数的 bf16 KV);`VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION` 改变 Q 是否量化的默认决策面,会影响 trtllm decode/prefill 内核的选型;对非 fp8 KV 用户无影响。

## 关联

同区域:pr-22703(FlashInfer TRTLLM FP8-q/kv NVFP4-out attention kernel)、pr-22675(flashinfer cubin 路径 env)、pr-21078(FlashInfer MLA trtllm-gen decode)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
