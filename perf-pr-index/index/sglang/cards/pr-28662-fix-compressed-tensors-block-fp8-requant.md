# PR #28662: [Fix] compressed-tensors block FP8: requantize weight scales to UE8M0 for DeepGEMM on Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Jiminator |
| 合并 | 2026-06-26 |
| 分类 | MoE(tier: major) |
| Label | quant, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/28662 |

## 一句话结论

修复 compressed-tensors block FP8 在 Blackwell 上的 NaN(权重 scale 未转 UE8M0),恢复 DeepGEMM 快路径。

## 动机

compressed-tensors block FP8(`CompressedTensorsW8A8Fp8` 的 BLOCK 策略)在 Blackwell(SM100)默认 GEMM 后端下产生 NaN / 空输出,例如 poolside/Laguna-M.1-FP8 的 GSM8K 为 0.00%、所有生成内容为空。Blackwell 上 block-FP8 linear 分派到 DeepGEMM,它把 activation 量化为 UE8M0 scale,并期望 weight scale 也是 UE8M0 打包。`Fp8LinearMethod` 已在 DeepGEMM runner 下做 requant,但 compressed-tensors 的 BLOCK 分支从未做——保留了 raw float32 block scale,DeepGEMM 将 UE8M0 activation scale 与 float32 weight scale 组合后输出 NaN。

## 优化点

在 `CompressedTensorsW8A8Fp8.process_weights_after_loading` 的 BLOCK 分支补上 UE8M0 requant;把 requant 判定抽取为共享 helper `requant_block_scale_ue8m0_for_deepgemm()`(`fp8_utils.py`),`Fp8LinearMethod`、`Fp8MoEMethod` 与 compressed-tensors 三个调用点统一走它,门控条件(DeepGEMM runner 被选中 + 128x128 block + bf16/aligned shape + 尚未是 UE8M0 + `should_deepgemm_weight_requant_ue8m0()`,即 SM100 + `DEEPGEMM_SCALE_UE8M0`)集中一处,替代原先在 `fp8.py` 中重复的判定代码。非 Blackwell 硬件与 `triton`/`cutlass` 后端保持 float32 scale 不变;属于行为保持的重构 + 缺陷修复,无新 kernel,稳态 GEMM 不变。

## 关键改动

```text
python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py: BLOCK 分支补 requant
python/sglang/srt/layers/quantization/fp8_utils.py: 新增共享 helper requant_block_scale_ue8m0_for_deepgemm
python/sglang/srt/layers/quantization/fp8.py: Fp8LinearMethod/Fp8MoEMethod 改走共享 helper
test/registered/unit/layers/quantization/test_deepgemm_ue8m0_requant.py: CPU 门控单测(mock)
test/manual/quant/test_compressed_tensors_fp8_block_ue8m0.py: Blackwell 端到端回归测试
```

## 性能声明

poolside/Laguna-M.1-FP8、8x B200、`--tp 8`、默认 GEMM 后端:GSM8K(1319, greedy)修复前 0.00%(空输出)→ 修复后 92.80%(H200 FP8 参考值 93.25%)。修复后默认 DeepGEMM 路径 vs `--fp8-gemm-backend triton` 回退(bench_serving random 1024 in / 512 out,cc=32):output 1363.6 vs 1084.2 tok/s(+25.8%)、single-stream decode 103.6 vs 67.0 tok/s(+54.8%)、single-stream TPOT 9.59 vs 14.87ms;PR 强调无新 kernel,收益纯粹来自恢复 DeepGEMM 默认路径。

## 验证与回归风险

CPU 注册单测覆盖门控各分支(bf16 输出、DeepGEMM 不支持的 shape、非 DeepGEMM runner、非 128x128 block、幂等只触发一次);Blackwell-only 手动回归测试断言 scale 被requant 且输出有限、接近 bf16 dequant 参考(修复前为 NaN)。NVFP4(CUTLASS 路径)与 compressed-tensors FP8 MoE experts 不受影响(不走 DeepGEMM)。

## 关联

同分类(MoE)另有 PR #19003;同类 FP8 scale 处理另见 PR #3148(Quantization)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
