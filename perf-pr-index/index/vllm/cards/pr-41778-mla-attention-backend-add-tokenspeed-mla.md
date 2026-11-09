# PR #41778: [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-05-14 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, performance, ready, ci/build, v1, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 可用但被截断(3 个文件未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/41778 |

## 一句话结论

把 tokenspeed_mla CuTe DSL 内核接入 vLLM 作为新的 MLA 后端(Blackwell + FP8 KV + DS R1 尺寸),生产批量 decode 提速 1.3–1.5×、MTP spec decode 2–4×。

## 动机

Blackwell 上 MLA decode 缺少针对 decode+MTP 场景优化的内核:现有 FlashInfer MLA / trtllm 在生产批量(bs≥32)与 spec decode(MTP-3+)下不是最优。TokenSpeed 的 CuTe DSL kernel 对 FP8 Q + FP8 KV 形状特化(DeepSeek R1 MLA 维度 qk_nope=128、qk_rope=64、v=128),KV 扫描随总 KV 量亚线性扩展,而 trtllm 大致线性,长上下文大批量下差距显著,值得作为可选后端接入,默认排在 FLASHINFER_MLA 之后。

## 优化点

新增两个后端文件:decode 侧 `TokenspeedMLABackend/Impl`——要求 SM100、FP8 KV cache(HND 布局)、R1 尺寸,借助 `supports_quant_query_input` 让上游把 Q concat+FP8 量化,`forward_mqa` 把 q 重排为 (num_decodes, q_len, heads, dim) 后调 `tokenspeed_mla_decode`,softmax_scale/output_scale 乘入 q_scale/k_scale 以从 FP8 KV(V 存为 V_real/k_scale)还原输出,lazy 分配 workspace(num_sms×heads×MAX_Q_LEN=8×(kv_lora_rank+1)×4B);prefill 侧 `TokenspeedMLAPrefillBackend` 注册进 MLA prefill registry/selector(SM100 优先级列表末位),处理 chunked context 与 LSE 转置,并对 v 强制 contiguous 避免内核内静默拷贝;启动时对 bf16/fp8 两种 q dtype 预 JIT warmup。CUDA 平台默认优先级中 TOKENSPEED_MLA 插在 FLASHINFER_MLA 之后(bs≈8 以上占优,bs≤2 回退)。

## 关键改动

```text
vllm/v1/attention/backends/mla/tokenspeed_mla.py: 新增 decode 后端(backend/impl/metadata builder)
vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py: 新增 prefill 后端(注册进 registry/selector)
vllm/platforms/cuda.py: SM100 MLA 默认优先级插入 TOKENSPEED_MLA(FLASHINFER_MLA 之后)
vllm/model_executor/layers/attention/mla_attention.py: backend_supports_prefill_query_quantization 加入 TOKENSPEED_MLA
tests/v1/attention/test_mla_backends.py: 后端正确性测试纳入 TOKENSPEED_MLA 与 prefill 后端参数化
```

decode 尺度/量化还原:

```python
self.softmax_scale = self.scale * layer._q_scale_float * layer._k_scale_float
self.output_scale = layer._k_scale_float
o = tokenspeed_mla_decode(query=q, kv_cache=kv_c_and_k_pe_cache, ...)
```

## 性能声明

微基准(Blackwell,FP8 KV,DS R1 维度;decode 对比 `trtllm_batch_decode_with_kv_cache_mla`,prefill 对比 `trtllm_ragged_attention_deep_seek` 与 fa4;50 repeats×10 warmup 或标注):

- decode MTP-3(q_len=4,kv=80k):nh=16 时 bs=64 达 2.33×(1420→610 µs),bs=8 起稳定 >1.6×;nh=32 时峰值 4.11×(bs=64,2596→631 µs)。MTP-7 峰值 4.23×。超过 crossover(bs≈3-5)后到 bs=512 保持 1.8–3.9×,不再回到持平。
- decode 单用户/小批量(bs≤2-4)回退 0.56–0.79×(约 110 µs 启动开销),KV 短时亦回退。
- benchmark.py 扫描(nh∈{16..128}×bs×kv∈{4k,16k}):nh=32、kv=16k 时 1.35–1.49×;nh=32、bs≥64、kv=4k 时 1.36–1.37×;nh=128 大多 ≈1.0×。
- prefill 短中上下文回退 2–5×(~120 µs 固定 CuTe-DSL 启动开销);≥64k 上下文接近持平(0.90–0.98×);q64k 时 fa4 120,592 µs vs TS 134,174 µs(0.90×)。
- 官方建议:生产 decode(bs≥32, nh≤32)用 TOKENSPEED_MLA(1.3–1.5×);decode+MTP-3+ 用之(2–4×);纯 prefill <16k 用 FA4/TRT-LLM。
- GSM8K(Kimi NVFP4 checkpoint):93.4。

## 验证与回归风险

`tests/v1/attention/test_mla_backends.py` 把 TOKENSPEED_MLA 加入 BACKENDS_TO_TEST(未安装 tokenspeed_mla 包则跳过)并新增 PREFILL_BACKENDS_TO_TEST 参数化(经 validate_configuration 按能力/依赖/R1 尺寸过滤);MockMLAAttentionLayer 改为继承 MLAAttention 以满足按 isinstance 过滤 static_forward_context 的 prefill 后端;decode scale 统一为 prefill_scale(1/sqrt(qk_head_dim))。回归风险:改变 SM100 MLA 默认优先级顺序(装了 tokenspeed_mla 且满足 R1 尺寸+FP8 KV 时会自动选中),不满足时经 supports_combination 拒绝并回落;UNIFORM_BATCH cudagraph 支持,query_len 需 UNIFORM;alibi/sliding window/soft cap 不支持。

## 关联

同区域:pr-21078(FlashInfer MLA trtllm-gen decode)、pr-20769(SM100 CUTLASS MLA decode)、pr-48642(sparse MLA fp8);prefill 注册机制对应 MLAPrefillBackendEnum(TRTLLM_RAGGED/FLASHINFER)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
