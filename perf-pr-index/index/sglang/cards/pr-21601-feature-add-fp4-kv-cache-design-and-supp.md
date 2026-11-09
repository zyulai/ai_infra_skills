# PR #21601: [Feature] Add FP4 KV Cache Design and support SM120 GPUs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @samuellees |
| 合并 | 2026-07-17 |
| 分类 | Quantization |
| Label | documentation, quant, blackwell, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计);diff 有省略(14 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/21601 |

## 一句话结论

新增 NVFP4(FP4 E2M1)KV cache 量化并支持 SM120,较 FP8 KV 省约一半显存,decode 提速 1.18x。

## 动机

FP8 KV cache 之后,长上下文场景需要更激进的 KV 显存压缩:NVFP4 packed FP4 存储可比 FP8 KV cache 再省约 2 倍内存,但 attention 的 prefill 与 decode 后端对 FP4 的消费方式不同,需要一套统一的设计来安全接入,同时把 hybrid GDN 模型的支持面扩展到 SM120(如 RTX PRO 6000 Blackwell)。

## 优化点

新增 `--kv-cache-dtype nvfp4`:quant_method(纯计算)、Pool(buffer + 批量 dequant)、Backend(视图适配)三方分工。核心是 KVCacheAttentionAccess 注册表——packed `torch.float4_e2m1fn_x2` 只描述存储,不说明 scale 语义,同一 recipe 在 prefill/decode 的 KV 视图也不同;注册表按 (phase, backend_name, tags) 声明 PLAIN / DEQUANT_WORKSPACE / NATIVE_FP4 三种访问方式,不支持的组合直接 fail fast。NVFP4 具体路径:prefill 走 FlashInfer dequant workspace——缓存前缀 FP4 反量化成 FP8、当前 chunk 由 bf16 转成 FP8,再用 FlashInfer FP8 paged prefill;decode 走 TRT-LLM MHA 原生 FP4——直接把 packed FP4 buffer 与 per-block scale(kv_cache_sf)交给 `trtllm_batch_decode_with_kv_cache`,BMM scale 由 quant method 提供(q_scale * k_scale * layer.scaling / v_scale)。decode 输出 dtype 对齐模型 dtype。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py: KVCacheAttentionAccess 注册表与 NVFP4 recipe 声明
python/sglang/srt/layers/attention/trtllm_mha_backend.py: 原生 FP4 decode(kv_cache_sf + BMM scales),prefill 报错回退
python/sglang/srt/layers/attention/flashinfer_backend.py: dequant workspace 的 page table/lens 元数据与 prefill 路径
python/sglang/srt/layers/attention/attention_registry.py: SM120 上 hybrid GDN 允许 triton/trtllm_mha/flashinfer 组合
```

## 性能声明

TPOT 类延迟对比(模型/硬件未逐项注明,表中为 FP8 KV vs NVFP4 KV):Prefill Latency 160K 上下文 8757 ms 对 8792 ms(0.996x);Prefill Latency 1M 上下文 142143 ms 对 142325 ms(0.998x);Decode Latency 1M 上下文 8.4 ms 对 7.1 ms(1.18x)。准确性:Qwen3.5-35B-A3B-FP8 模型 NVFP4 KV,与 BF16/FP8 KV 对比——GSM8K 1319 题 97.18%(BF16 97.10%,FP8 96.95%);GPQA 8 轮均值 83.33%(BF16 82.26%,FP8 83.08%);LongBench V2 128K 子集 52.70%(BF16 52.03%,FP8 未测)。LLaMA-3.1-8B:GSM8K NVFP4 63.85% 对 FP8 64.6%(注:素材正则把 63.85% 误标为吞吐,实为准确率)。

## 验证与回归风险

依赖 `resolve_kv_cache_quant`、MHATokenToKVPool 的 packed FP4 buffer 与 FP8 workspace、`get_raw_fp4_kv_buffer`/`get_flashinfer_dequant_workspace_kv_buffer` 等新 pool 接口;TRT-LLM MHA 路径 decode-only,与 flashinfer/triton prefill 后端组合使用。回归面:flashinfer backend 的 kv_cache_dtype 判定改为 flashinfer_kv_cache_dtype(dequant workspace 时固定 fp8_e4m3fn),触及非 FP4 用户的 tensor-core 判定;attention_registry 对 SM120/其他 Blackwell 的 backend 白名单拆分;sliding window 与 cross-attention 路径显式拒绝 custom_kv_indices。

## 关联

Roadmap:sgl-project/sglang issue #29913;前序 PR #18314(本 PR 由其 rebase 而来);同区域:FP8 KV cache(fp8_e4m3/e5m2)与 fp4_mx_block16 recipe。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
