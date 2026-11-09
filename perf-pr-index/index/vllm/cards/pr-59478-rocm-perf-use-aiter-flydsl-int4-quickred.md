# PR #59478: [ROCm][Perf] Use AITER FlyDSL INT4 QuickReduce when INT4 QuickReduce is selected

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-10-26 |
| 分类 | Quantization |
| Label | rocm, ready, torch.compile |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/59478 |

## 一句话结论

选 INT4 QuickReduce 时,bf16 对齐 payload 改走 AITER FlyDSL QuickAllReduceInt4:kernel 最高 2.96× 提速,TP8 服务吞吐 geomean +4.4%。

## 动机

gfx942/gfx950 上选择 `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` 时,vLLM 只有 HIP QuickReduce 一条路。AITER 已合入 FlyDSL 实现的 `QuickAllReduceInt4`(ROCm/aiter#4970,支持 TP 2/4/8),官方测量对 bf16 payload 比 HIP INT4 快 29–47%,vLLM 此前未接入。FlyDSL 尚无 fused RMSNorm kernel,若 `fuse_allreduce_rms` 开着,这些 payload 只能走 all-reduce + 独立 RMSNorm;PR 实测该组合并不慢,并一次性 warn 说明 fusion 对其无效。

## 优化点

`QuickAllReduce` 初始化时,若 quant level 为 INT4 且 `is_flydsl_quick_reduce_available`(CDNA3/CDNA4)成立,则额外创建 FlyDSL communicator,并用 1 MiB bf16 warmup 提前 `compile()`,把 JIT 编译移出首个 prefill 与 CUDA graph capture。`quick_all_reduce` 新增 `_use_flydsl_int4` 判定:bf16、contiguous、16 字节对齐,且按 HIP 窗口落在 INT4 档;命中即走 FlyDSL `allreduce`,否则 HIP kernel 原样执行。HIP QuickReduce 的 size window(TP4/8 ≥2 MiB、TP2 ≥1 MiB)优先且不放宽——0.5 MiB 时 FlyDSL 比 custom-AR 慢 14%。`fuse_allreduce_rms` pass 对 FlyDSL payload 跳过 fused kernel,改走 `_unfused_allreduce_rmsnorm`;`close()` 同步释放 FlyDSL communicator。无新增 env var。

## 关键改动

```text
vllm/distributed/device_communicators/quick_all_reduce.py: FlyDSL 初始化/路由/释放
vllm/_aiter_ops.py: is_flydsl_quick_reduce_available(CDNA3/4)与工厂函数;fused AR+RMSNorm 的 unfused 回退
tests/distributed/test_quick_all_reduce.py: dispatch 回退与 close 测试
tests/distributed/test_aiter_rms_fallback.py: FlyDSL payload 不得进 fused kernel
```

```python
def _use_flydsl_int4(self, inp: torch.Tensor) -> bool:
    return (
        self._flydsl_int4 is not None
        and inp.dtype == torch.bfloat16
        and inp.is_contiguous()
        and inp.data_ptr() % 16 == 0
        and self._get_qr_quant_level(inp) == QuickReduceRegime.INT4.value
    )
```

## 性能声明

- AITER 上游测量:TP8 下比 HIP INT4 快 29–47%(512×4096:18.1 vs 26.5 µs;32768×5120:558 vs 1061 µs)。
- Kernel routing(MI355X,TP4):bf16 2 MiB 24.0 vs 30.4 µs、256 MiB 767.8 vs 938.9 µs;bf16 payload 整体 1.2–1.5× 快于 HIP kernel;fp16 仍走 HIP。
- fused AR+RMSNorm 对比(nightly 路线 vs FlyDSL + 独立 RMSNorm):TP2 8 MiB 2.68×、TP4 32 MiB 2.53×、TP8 32 MiB 1.85×,整体 1.05–2.96×。
- Serving(amd/Qwen3.8-2.4T-A95B-Quark-MXFP4,TP8,MI355X,mbt 8192,INT4):九 cell 吞吐 geomean 1.044× stock(18 个 seed 对中 16 个领先),中位 TPOT 0.963×;prefill-heavy cell 多为 +4~6%,1024/1024 c128 持平;GSM8K 0.9726 → 0.9726/0.9749 不变。

## 验证与回归风险

新增 `test_flydsl_int4_dispatch_falls_back_to_hip`(fp16/非连续/未对齐/FP-codec payload 必须落 HIP)、`test_close_closes_flydsl_int4`、`test_rocm_aiter_fused_rmsnorm_skips_fused_kernels_for_flydsl_payloads`;pytest 45 passed / 1 skipped / 2 failed,两个失败在无 PR 的 stock nightly 上同样复现(既有问题,与 INT4/INT3 graph 用例的量化误差相关)。风险:serving A/B 的 stock 臂只跑一次、无 A/A control,32768/128 c4 cell 两种子间波动大(1.07/1.17);收益集中在 ≥2 MiB payload(prefill 与 ≥128 token 的 decode batch),更小 payload 不受影响;`fuse_allreduce_rms` 对 FlyDSL payload 实际失效;被测 nightly 的 fused HIP QR+RMSNorm 因 aiter `qr_get_handle` 签名变更本就静默回退,PR 未修该函数。

## 关联

上游 ROCm/aiter#4970(FlyDSL `QuickAllReduceInt4`);后续 vLLM PR #61021(RMSNorm epilogue)、#61020(MXFP4 epilogue)在其上加融合,需 ROCm/aiter#6369。索引同区域另有 pr-32520(FP4 量化 kernel 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
