# PR #44114: [Perf] [NVIDIA] qsa: Keep sparse decode scratch in FP8 for FP8 KV pools

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-10-17 |
| 分类 | Quantization(major) |
| Label | run-ci, bypass-fail-fast |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/44114 |

## 一句话结论

QSA sparse decode 在 FP8 KV pool 下把 packed scratch 保留为 FP8,单步解码时延降 3.76%,服务吞吐提升约 1.5%–2.3%。

## 动机

#38851 修复 QSA sparse-decode corruption 时做了三件事:zero-fill scratch 尾部、int64 pool offsets、把 FP8 pool 行 dequant 成 BF16 scratch。前两项是修 bug,第三项是保守选择——它让 TRTLLM-gen decode 始终跑 BF16-q + BF16-KV。FlashInfer 的 `trtllm_batch_decode_with_kv_cache` 本身支持 BF16-q + FP8-E4M3-KV,因此 FP8 pool 下把 packed scratch 保持在 FP8 可以把 gather 与 attention 的 KV 流量减半,并去掉 `fused_kv_prepare` 中的 dequant。

## 优化点

改动集中在 `_forward_trtllm_sparse`:当 KV pool 为 `float8_e4m3fn` 且设备为 SM10x 时,packed K/V scratch 直接按 pool dtype(FP8)分配,否则维持按 query dtype(BF16)分配的旧行为;query 始终为 BF16,TRTLLM-gen 走 BF16-q + FP8-E4M3-KV 路径;scratch cache 本已按 dtype 分 key。#38851 的 zero-fill、int64 offsets 与 scale-free 处理不变;BF16 pool、FA2/FA4 fallback、prefill、SM120 均不变(SM120 仍用 BF16 scratch;该路径数值上只在 SM100 与 SM120 验证过,不能扩大到所有 SM12x 设备,否则会在 SM121/GB10 上静默损坏长上下文 decode)。

## 关键改动

```text
python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py: _forward_trtllm_sparse 按pool dtype+SM10x 选 packed_dtype;is_sm100_supported/is_sm120 提升到模块级供测试 monkeypatch
test/registered/kernels/ops/attention/qsa/test_qsa_fused_kv_prepare.py: 参数化 (dtype, packed_dtype),新增 FP8 pool -> FP8 scratch 组合
test/registered/kernels/ops/attention/qsa/test_qsa_strided_zero_fill.py: 同上,断言按 packed_dtype 比较
```

## 性能声明

Qwen3.8-Flash-Next-NVFP4,VR200,TP1,FP8 KV,FP8 QSA indexer,NEXTN MTP(3 steps,4 draft tokens),C64 with MTP。每 decode 步:`fused_kv_prepare` 135.19→86.69 µs,FMHA 97.61→89.42 µs,步时延 20.686→19.908 ms(-3.76%)。Agentic multi-turn serving,3 组配对 1 小时运行:median per-user throughput(P90)+1.49%,median total throughput per GPU +2.29%。作者推算:decode 约占 wall time 68%,-3.76% 每步约对应 +2.6% E2E。

## 验证与回归风险

精度:同配置采样(temperature 1.0、top_p 0.95、top_k 20、max_new_tokens 131072,无样本触顶),AIME24+25(60×16)90.52%→92.08%(+1.56 pp,95% CI [-0.47, +3.59]),GPQA(100×8)87.25%→88.62%(+1.38 pp,SE 1.09),GSM8K(200)两侧均 198;输出长度变化 +0.15%;对 FP32 参考的 attention 输出误差两条路径均约 2.1e-3,且 KV 长度到 64K 以上不增长。单元测试扩展覆盖 FP8 pool → FP8 scratch 的 prepare 与 zero-fill 尾部。风险面:arch gate 依赖 `is_sm100_supported()`,SM120 与非 SM10x 维持旧路径;错误扩大 arch 范围会在 SM121/GB10 上静默损坏长上下文 decode(PR 注释明确警示)。

## 关联

#38851(前序:QSA sparse-decode corruption 修复,本 PR 移除其中保守的 dequant 选择)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
