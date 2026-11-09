# PR #7798: [Core][Kernels] Use FlashInfer backend for FP8 KV Cache when available.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pavanimajety |
| 合并 | 2024-08-28 |
| 分类 | Quantization |
| Label | ready |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/7798 |

## 一句话结论

`--kv-cache-dtype=fp8` 时接入 FlashInfer 后端,缓解 Xformers fp8 paged attention 的性能塌陷,H200 上 decode 提速约 20%。

## 动机

`--kv-cache-dtype=fp8` 下 vLLM 默认回落到 Xformers 后端的 `paged_attention_v1` fp8 kernel,吞吐显著劣化:FP8 量化 + FP8 KV cache 相对 FP8 量化 + FP16 KV cache 落后约 30%。FlashInfer 自 v0.1.4 起 paged KV cache 原生支持 fp8(e4m3/e5m2,带 k_scale/v_scale),值得接通。

## 优化点

(1) `FlashInferBackend.get_fp8_dtype_for_flashinfer`:把 vLLM 的 kv_cache_dtype 字符串(fp8/fp8_e4m3→`torch.float8_e4m3fn`,fp8_e5m2→`torch.float8_e5m2`)映射为 torch fp8 dtype,graph capture 与 eager 两条路径统一用其确定 kv_cache dtype。(2) `forward()` 中当 KV cache 为 fp8 时,把 `reshape_and_cache_flash` 写入的 cache `view(torch_dtype)` 成 fp8 视图再传给 FlashInfer decode/prefill wrapper,并把 `k_scale`/`v_scale` 透传给 `wrapper.forward`(反量化在 kernel 内完成)。(3) `selector.py`:FP8 KV cache 回退 Xformers 时新增 warning,提示设 `VLLM_ATTENTION_BACKEND=FLASHINFER` 获得更好性能(该后端仍需显式 env 选择)。另修正 decode 测试的 `use_tensor_cores` 判定(GQA 比率 >4 才用 tensor core)。

## 关键改动

```text
vllm/attention/backends/flashinfer.py: fp8 dtype 映射 + kv_cache 视图转换 + k/v_scale 透传
vllm/attention/selector.py: FP8 KV cache 下提示改用 FLASHINFER 后端
tests/kernels/test_flashinfer.py: 新增 fp8 KV prefill/decode 对比参考实现测试
```

```python
# The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
# to process the cache in fp8
torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
    self.kv_cache_dtype)
kv_cache = kv_cache.view(torch_dtype)
...
output = flashinfer_attn_with_kv_cache(..., k_scale=k_scale, v_scale=v_scale)
```

## 性能声明

H200:相对 Xformers 后端 fp8 `paged_attention_v1` kernel,约 20% perf improvement。

`benchmark_throughput.py`,`neuralmagic/Meta-Llama-3-70B-Instruct-FP8-KV`,1024 prompts,input/output 各 256,TP=1(吞吐单位 PR 未注明):

| 后端 | FP8 量化 + FP16 KV | FP8 量化 + FP8 KV |
|---|---|---|
| xformers | 2606.275 | 1792.68(较自身 FP16 KV 落后约 30%) |
| flashinfer | 2340.92 | 2210.09(较 Xformers FP16 KV 落后约 15%) |

即 fp8 KV cache 下的吞吐由 1792.68 提升至 2210.09;但 FP16 KV cache 下 FlashInfer(2340.92)仍低于 Xformers(2606.28),因此 FP8 KV cache 场景才建议 FlashInfer。

## 验证与回归风险

新增 `test_flashinfer_prefill_with_paged_fp8_kv` 与 `test_flashinfer_decode_with_paged_fp8_kv`:构造 fp8 KV cache(按 amax/448 求 k_scale/v_scale),与 fp16 参考实现对比,decode 因 FlashInfer 精度问题放宽为 atol 2e-2。风险面:仅覆盖 FlashInfer 后端且 chunked prefill 与 FlashInfer 不兼容(断言保持);fp8_e5m2 路径按同接口支持但性能未单独报告;decode 测试注明"Temporary fix: Seems like a flashinfer issue";selector 仍默认 Xformers,需 env 显式启用该行为。

## 关联

同区域(FP8 KV cache / FlashInfer attention):索引内 PR #21078(FlashInfer MLA TRTLLM-gen decode)、PR #20034(FlashInfer MLA ragged prefill)、PR #38479(TurboQuant 2-bit KV cache)、PR #38445 系列;FlashInfer 后端在 vLLM 的早期接入点之一,后续 backend 选择逻辑演化为 #52861 等。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
