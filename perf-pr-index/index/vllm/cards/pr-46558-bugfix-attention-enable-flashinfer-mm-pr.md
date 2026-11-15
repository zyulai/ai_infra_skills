# PR #46558: [Bugfix][Attention] Enable FlashInfer mm-prefix attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lesj0610 |
| 合并 | 2026-11-11 |
| 分类 | Attention / Kernel(tier: major) |
| Label | bug, documentation, ready, v1, multi-modality, nvidia |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/46558 |

## 一句话结论

让 multimodal PrefixLM batch 留在原生 FlashInfer backend:接入 FlashInfer 的 mm-prefix batch-prefill wrapper(每 KV tile 评估 `(causal AND sliding_window) OR mm_prefix`),不物化 `O(qo_len * kv_len)` mask——range buffer 只占每 scheduled token 8 字节(16,384 tokens 时 128 KiB);节省的是内存而非 FLOPs(prefill 仍遍历整个矩形)。

## 动机

此前 mm-prefix(PrefixLM)batch 无法由原生 FlashInfer 服务(#47118 只对 text-only 请求启用 FlashInfer),需要 composite 或 fallback 路径。wrapper 来自 FlashInfer #5189(0.7.1 起随附,main 的 pin 已含),NVFP4 的 `v_sf_swizzled` 选项来自 FlashInfer #7012(0.7.3,main 自 #65930 起 pin 该版本)。

## 优化点

- **Adapter**:`FlashInferMetadataBuilder` 从 request metadata 构建每 query 的 `[start, end]` 行(`fill_mm_prefix_query_ranges`),`int32 (N, 2)` buffer 按 scheduled tokens 定容;一个谓词 `_native_mm_prefix` 同时决定 backend 选择与 builder 的 range 状态,composite 的 FlashInfer child 保持 main 的路由、不分配 range buffer。
- **路由**:mm-prefix wrapper 与常规 prefill path 并列规划——text-only batch、decode 与 forward 时 range 被清空的层(如 Gemma 4 full-attention groups)继续走 trtllm-gen 或 FA3;SM100 家族自动选择仍是 `TRITON_FLASHINFER`,显式选 `FLASHINFER` 才原生服务 range 且不 advertise large pages;SM90 自动优先级(偏好 `TRITON_FLASH_ATTN` composite)不变。SM100 NVFP4 cache 的 V scale 在 trtllm-gen swizzle 中,只有 wrapper 接受 `v_sf_swizzled` 时才接受,否则拒绝。
- **FP8 cache**:`fp8`/`fp8_e4m3` cache 也走原生路径——fa2 无 FP8 query/output,只适配 mm-prefix call:FP8 query 用 q_scale 无损回模型 dtype,fused FP8 output 在 call 后量化;原生 mm-prefix 模型只提供 FP8 output fusion(与 composite 一致)。
- **拒绝项**:FlashInfer 在选择时拒绝 `fp8_e5m2`、`nvfp4_4over6`、sinks 与 DCP,engine 启动时拒绝 `logits_soft_cap`。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: _MM_PREFIX_KV_DTYPES / _MM_PREFIX_READS_SWIZZLED_V_SF / _native_mm_prefix 谓词;supports_mm_prefix / supports_combination 扩展;FlashInferMetadata 增加 mm_prefix_query_range_tensor / mm_wrapper / mm_prefill_ranges;builder 构建 ranges 并 plan mm wrapper;forward 增加 mm-prefix 分支(含 FP8 query 还原与 output 量化)
tests/kernels/attention/test_attention_selector.py: mm-prefix backend 选择与 NVFP4 swizzled gate 测试
tests/models/multimodal/generation/test_mm_prefix_lm.py: e2e 测试参数化 attention_backend=[None, "FLASHINFER"]
```

```python
+def _native_mm_prefix(vllm_config: VllmConfig | None) -> bool:
+    """Whether FlashInfer serves this model's mm-prefix ranges itself; on SM100
+    only when FLASHINFER is selected explicitly."""
+    from vllm.v1.attention.backends.registry import AttentionBackendEnum
+
+    if vllm_config is None or vllm_config.model_config is None:
+        return False
+    model_config = vllm_config.model_config
+    return (
+        model_config.is_mm_prefix_lm
+        # The DCP prefill's context/new-token split breaks the ranges.
+        and vllm_config.parallel_config.decode_context_parallel_size == 1
+        # The wrapper has no logits soft cap (e.g. Gemma 2 language models).
+        and not getattr(model_config.hf_text_config, "attn_logit_softcapping", None)
+        and (
+            not vllm_config.cache_config.cache_dtype.startswith("nvfp4")
+            or _nvfp4_kv_on_fa2()
+            or _MM_PREFIX_READS_SWIZZLED_V_SF
+        )
+        and (
+            not current_platform.is_device_capability_family(100)
+            or vllm_config.attention_config.backend == AttentionBackendEnum.FLASHINFER
+        )
+    )
```

## 性能声明

- 本 PR 当前实现只有功能/正确性验证,无端到端加速声明;正文的量化结果为两类:
  - **sm_103 显式 `FLASHINFER` vs `TRITON_FLASHINFER`**(RedHatAI/gemma-4-12B-it-NVFP4 + FP8 KV cache,每配置 1 run,代码等价于 37c1fa3980、cleanup 前):text throughput 在并发 32 低 0.2%;image throughput 并发 1 低 0.8%、并发 32/128 分别高 0.5%/1.6%;ChartQA(500 examples,relaxed accuracy)68.2 vs 66.8。正文注明"These numbers describe this run only";compiler-fused FP8 output 路径未在完整模型上验证。Gemma 3 4B NVFP4 KV(TRITON_FLASHINFER 在 SM100 拒绝)ChartQA 43.8 vs FP8 KV 的 43.6。
  - **历史 serving 数据**(正文明确标注为"historical context",来自更早的 in-tree 变体,不构成对当前 wrapper-based adapter 的验证):2×SM80、-tp 2、31B 多模态模型,TRITON_ATTN / FLASHINFER——并发 1 total tok/s 14.2 / 58.5、TTFT 3884 / 2167 ms;并发 32 output tok/s 193.3 / 188.4(TRITON_ATTN 为单 seed,其 2.6% 领先有未测 seed 方差);并发 128 183.7 / 193.4。注意事项:两臂 KV 预算不同(FlashInfer 配置约少 3.7% KV 容量)、比较的是完整 serving 配置而非隔离 attention kernel。
- 正文对机制的定义:"the gain is memory, not FLOPs"——range buffer 为 `int32 (N, 2)`,16,384 tokens 时 128 KiB,无 dense mask。

## 验证与回归风险

- sm_80 + FlashInfer 0.7.2:`tests/v1/attention/test_mm_prefix.py tests/v1/attention/test_attention_splitting.py -rs` 24 passed/22 skipped(FA4 mask_mod);selector 测试 25 passed(Hopper/Blackwell case skipped);Gemma 3 E2E 在自动 backend 与显式 `FLASHINFER` 下均通过。selector 测试覆盖 SM100 自动 composite 与显式 `FLASHINFER`(no-quant 与 FP8 KV)及 NVFP4 有/无 `v_sf_swizzled`。cleanup 前(37c1fa3980 等价代码)曾在 sm_80/sm_103 用 dense fp32 reference 跑过 GPU 测试(含 FP8 KV),review 中被 E2E 测试替换。
- pre-commit 通过。
- 风险:mm-prefix wrapper 的 `plan()` 不收 seq_lens,kv_lens 在 pageable memory 构建、以 `non_blocking=True` 拷贝,代码中以 `gpu_sync_allowed()` WORKAROUND 标注(待 wrapper 支持 pinned seq_lens 后移除);compiler-fused FP8 output 路径未在完整模型验证;sm_103 的对比数据每配置仅 1 run;wrapper 的 JIT 模块在构造时编译。

## 关联

- FlashInfer #5189(mm-prefix wrapper)、#7012(`v_sf_swizzled`,随 0.7.3);#47118(Gemma 4 多模态的 text-only FlashInfer);#65930(main pin FlashInfer 0.7.3)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
