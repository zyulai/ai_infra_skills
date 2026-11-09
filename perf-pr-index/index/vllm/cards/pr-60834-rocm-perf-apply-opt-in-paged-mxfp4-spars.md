# PR #60834: [ROCm][Perf] Apply opt-in paged MXFP4 sparse indexer dense path for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-10-23 |
| 分类 | Quantization（tier: major） |
| Label | rocm, deepseek, DSv4, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/60834 |

## 一句话结论

为 DeepSeek-V4 在 ROCm 启用 opt-in 的 paged MXFP4 indexer dense path，agentic replay 下 mean TPOT -5.62%、KV cache 容量 +9.17%。

## 动机

PR #58671 已为 DeepSeek-V4.1 在 ROCm gfx950 加入 paged MXFP4 sparse indexer：indexer K cache 以 MXFP4 存储（68 B/token，FP8 为 132 B），按 preshuffled page 排布供 aiter `pa_mqa_logits_mxfp4` 就地打分，prefill 不再先 gather K，其中 dense path 是 model-agnostic 的。本 PR 把该 dense path 带到 DeepSeek-V4（如 V4-Pro：30 个 indexer 层、compress ratio 4、64 heads × 128、top-k 1024）。此前 V4 在 ROCm 只能用 FP8 indexer cache：`dsa_indexer_uses_fp4()` 在 ROCm 仅放行 `deepseek_v41`，`SparseAttnIndexer.forward_hip` 断言 cache 非 FP4，且 V4 自身 MXFP4 Q/K Triton kernel 依赖 CUDA PTX，无法在 ROCm 运行。特性与 V4.1 一样是 opt-in，默认 FP8 路径不变。

## 优化点

K 写入拆为两次 launch：新 Triton kernel `_compress_pool_kernel` 做 softmax-gated overlap pooling，把 boundary token 的一行 bf16 写入所有 indexer 层共享的 scratch buffer；再由 aiter `k_norm_rope_mxfp4_cache(..., shuffle=layout)` 经 `rocm_mxfp4_indexer_k_store` 完成 norm→RoPE→MXFP4 并存入 preshuffled pages，页布局与量化仍只由 aiter 定义。代价是每层多一次小 launch 与一次 bf16 round trip，舍入误差约 2⁻⁹，比 MXFP4 量化步低两个数量级。Q 复用 aiter `q_rope_mxfp4_quant`，产出折叠了 softmax 与 head scales 的 fp32 weights。打分与 metadata 原样复用 `RocmSparseAttnIndexer`（无 candidate blocks）和 `RocmMxfp4IndexerMetadataBuilder`，新增 `DeepseekV4RocmMxfp4IndexerBackend` 继承 `DeepseekV4IndexerBackend` 以沿用 256-token blocks 与 ROCm adaptive-verification，ratio 4 得到 64-entry pages（aiter 首选页大小）。共享 V4 代码仅加 extract-method / ClassVar hooks，代码原样搬移，CUDA 分支不受影响。

## 关键改动

```text
vllm/models/deepseek_v4/amd/rocm.py: 新增 DeepseekV4RocmMxfp4Compressor（_compress_pool_kernel + aiter cache op 两段式写 K）与 DeepseekV4RocmMxfp4IndexerBackend
vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py: 从 V4.1 迁入并共享 rocm_mxfp4_indexer_k_store / rocm_mxfp4_indexer_q_quant / _aiter_indexer_cache_ops
vllm/v1/attention/backends/mla/indexer.py: dsa_indexer_uses_fp4() 的 ROCm 分支新增接受 deepseek_v4
vllm/model_executor/warmup/jit_warmup_triton_helper.py: get_warmup_keys() 在 ROCm + MXFP4 cache 下返回空，避免 PTX 变体启动时编译失败
vllm/models/deepseek_v41/amd/rocm.py: cache ops 移出后仍保持可导入
```

## 性能声明

测试条件：DeepSeek-V4-Pro，ROCm gfx950，`VLLM_ROCM_USE_AITER=1`、`--attention-config '{"indexer_kv_dtype": "mxfp4"}'`，tp8，`--kv-cache-dtype fp8`，`--moe-backend aiter`，mtp speculative-config，SA InferenceX `8k1k` 固定序列长度与 `agentic` replay workload。固定序列长度（Main FP8 → This PR MXFP4）：conc=1 输出吞吐 145.99→156.34 tok/s（+7.09%）、mean TTFT 597.24→413.91 ms（-30.70%）、mean TPOT 6.24→5.96 ms（-4.49%）；conc=64 为 1629.53→1656.28（+1.64%）、TTFT -0.90%、TPOT -1.52%（conc=4/16 吞吐分别 +2.64%/+0.92%）。Agentic replay：吞吐 697.25→675.83 tok/s（-3.07%）、mean TTFT 2442.24→2321.05 ms（-4.96%）、mean TPOT 46.01→43.42 ms（-5.62%）、KV cache 容量 4,589,684→5,010,749 tokens（+9.17%）。PR 注明吞吐在噪声内、TTFT 同配置波动可达 13%，容量增益源于 68 B/token 相对 132 B/token。

## 验证与回归风险

单测：ROCm/CDNA4 平台门控的 `test_rocm_dsv4_mxfp4_indexer.py` 等，新测 4 passed（compressor pages 与 aiter writer 逐字节匹配 0 mismatched bytes，shifted window 对照 21.8% mismatch），V4.1 MXFP4 indexer 37 passed，V4 ROCm/compressor 99 passed、137 skipped。精度：GSM8K flexible-extract 0.9492→0.9477（-0.16%）、strict-match 0.9500→0.9484（-0.17%），GPQA Diamond 两者均 0.7121±0.0323。风险面：仅 ROCm gfx950 且 opt-in 生效，默认 FP8 路径不变；CUDA 分支未改；MXFP4 带来 <0.2% 的微小精度退化与每层多一次 launch 的开销。

## 关联

前序 PR #58671（为 DeepSeek-V4.1 引入 paged MXFP4 sparse indexer，本 PR 复用其 dense path、aiter cache op 与 metadata builder）；无其他明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
