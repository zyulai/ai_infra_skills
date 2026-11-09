# PR #26272: amd/deepseek_v4 33/N remove BF16-to-FP32 elementwise cast from compressor GEMM on HIP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-05-26 |
| 分类 | Attention / Kernel (tier: major) |
| Label | deepseek, jit-kernel |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用(2 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/26272 |

## 一句话结论

MI355/HIP 上让 compressor GEMM 直接输出 BF16,消掉每次迭代 90 个 BF16→FP32 cast kernel,预期 ITL +1.87%、e2e 吞吐 +2.1%。

## 动机

MI355/HIP 上 `Compressor.forward()` 调 `linear_bf16_fp32()`,即 `tgemm.mm(x, y, otype=x.dtype).float()`:MI355 的 aiter tuned flydsl 配置只支持 BF16 输出,`.float()` 会触发单独的 `bfloat16tofloat32_copy_kernel_cuda`(每次约 4.2 µs)。按层型统计:C 型层(compress_ratio=4)2 个 compressor GEMM 各 1 个 cast、B 型层(compress_ratio=128)1 个、A 型层 0 个,每迭代共 30x2+30x1=90 个 cast kernel,约 384 µs(约 ITL 的 1.87%)。B200 无此问题,因其 `deep_gemm`/`cublas` 路径原生单 kernel 输出 FP32。

## 优化点

在 HIP/aiter 路径上绕过 `linear_bf16_fp32()` 的 FP32 输出契约(其他路径如 cublas/deep_gemm/torch fallback 仍用),`Compressor.forward()` 直接调 `_tgemm.mm(x, self.wkv_gate.weight, otype=x.dtype)` 拿 BF16 输出,省掉 `.float()` round-trip(GEMM 内部本就用 FP32 累加,原路径是 FP32 累加→截断 BF16→cast 回 FP32→Triton 再 promote 到 FP32)。安全性:下游融合 Triton compress kernel(`_c4_decode_kernel`、`_c128_decode_kernel`、`_compress_norm_rope_kernel`)内部已经 `.to(tl.float32)` 提升;`CompressStatePool` 缓冲仍为 FP32(state_dtype=torch.float32),Triton 自动类型提升处理 BF16 存入 FP32 buffer。配套把 `fused_compress_triton.py` 中 `kv_score_input` 与 `out` 的 FP32-only 断言放宽为接受 BF16;`kv_score_buffer` 断言保持 FP32-only。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py: Compressor.forward() aiter 路径直调 _tgemm.mm 输出 BF16,绕过 .float()
python/sglang/srt/layers/attention/compressed/fused_compress_triton.py: _check_common 放宽 kv_score_input/out 的 dtype 断言
```

```python
if _use_aiter:
    kv_score = _tgemm.mm(x, self.wkv_gate.weight, otype=x.dtype)
else:
    kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
```

## 性能声明

Kernel 层(每 C 型层,decode):`hgemm_bf16`(C4 GEMM)8.8 µs + `bfloat16tofloat32_copy` 4.4 µs + `hgemm_bf16`(C128 GEMM)9.3 µs + cast 4.2 µs = 26.7 µs,改为 18.1 µs(cast 消除)。Kernel-to-E2E:C 型 30 层 x 8.6 µs + B 型 30 层 x 4.2 µs = 384 µs,预期 ITL 提升 384/20,590 = **1.87%**(约 0.38 ms);TTFT 影响可忽略(prefill 由大 GEMM 主导)。

E2E(il8k_ol1k,concurrency=4,32 prompts;DeepSeek-V4-Pro,TP=8,MI355,compressed attention backend,page size 256,chunked prefill 8192):总吞吐 1,558.2 → 1,591.2 tok/s(+2.1%),输出吞吐 173.1 → 176.8 tok/s(+2.1%),median ITL 20.59 → 20.46 ms(-0.6%),median TPOT 21.54 → 20.99 ms(-2.6%),median TTFT -0.7%,median E2E 延迟 -0.6%。

## 验证与回归风险

精度:GSM8K(2000 题,5-shot,parallel=1000)得分 0.908,阈值 0.88,PASS。Checklist 中 unit tests 与文档未勾选,diff 未新增测试。风险:数值路径变化(BF16 输出直接进 Triton compress kernel,少一次 BF16 截断,理论上与原路径不完全逐位一致,PR 未讨论);仅影响 HIP/aiter 的 DeepSeek V4 compressor 路径,非 aiter 路径不变;断言放宽后 `compress_dispatch` 现在接受 BF16 输入,其他调用方行为不变。

## 关联

amd/deepseek_v4 integration 分支的增量 PR,引用 Day-0 原始 PR #23600 与 #23608。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
