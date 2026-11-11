# PR #44486: [Perf] quantization: Add the bias in the cuBLASLt epilogue for per-tensor FP8 linears

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @gkswns0531 |
| 合并 | 2026-11-07 |
| 分类 | Quantization（tier: major） |
| Label | run-ci |
| Diff | +103/-20, 2 files（pack diffstat 缺失，按 pack diff 文本计数） |
| 链接 | https://github.com/sgl-project/sglang/pull/44486 |

## 一句话结论

带偏置的 per-tensor FP8 linear 在 ≥8192 行时改用 cuBLASLt epilogue 加 bias，bge-reranker 每次 rerank GPU 时间降 22–24%。

## 动机

自 #33148 起，per-tensor FP8 linear（`ModelOptFp8LinearMethod`）在 SM90 及以上走 FlashInfer `bmm_fp8`，而 `bmm_fp8` 不接受 bias，于是每个带偏置的 Linear 都要额外跑一次 `output + bias`：`BAAI/bge-reranker-v2-m3` 在 ModelOpt FP8 下、RTX PRO 6000 上每 batch 197 ms 中有 43 ms 花在这些加法上。

## 优化点

`apply_fp8_linear_bmm_flashinfer` 在有 bias 且行数 ≥8192 时改走 `torch._scaled_mm(..., bias=bias, use_fast_accum=True)`：同一个 cuBLASLt GEMM 在 epilogue 内加 bias，省掉单独的 add kernel（bmm_fp8 同样启用 fast accumulation）。8192 行以下路径不变——实测 SGLang autotune 过的 `bmm_fp8` 比未 autotune 的 `torch._scaled_mm` 更快，收益超过省掉的 add。阈值以常量 `_FP8_BIAS_EPILOGUE_MIN_ROWS` 固定；上游 flashinfer-ai/flashinfer#8811（in review）给 `bmm_fp8` 本身加 bias，届时该阈值可以移除。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_utils.py: 阈值常量与 torch._scaled_mm bias-epilogue 路径
test/registered/unit/layers/quantization/test_fp8_utils.py: epilogue 门控测试与数值一致性测试
```

```python
_FP8_BIAS_EPILOGUE_MIN_ROWS = 8192
...
if bias is not None and qinput.shape[0] >= _FP8_BIAS_EPILOGUE_MIN_ROWS:
    output = torch._scaled_mm(qinput, weight, out_dtype=output_dtype,
        scale_a=x_scale, scale_b=weight_scale, bias=bias, use_fast_accum=True)
else:
    output = flashinfer_bmm_fp8(qinput, weight, x_scale, weight_scale, output_dtype)
```

## 性能声明

RTX PRO 6000（SM120）、torch 2.14.1、FlashInfer 0.7.1，每次 `rerank` 调用的 GPU 时间（128 pairs，2 次均值）：bge-reranker-v2-m3 FP8 128×320 tokens 100.2→77.9 ms，128×640 tokens 196.6→150.3 ms。形状扫描（12 shapes × FP16/BF16）：≥8192 行时 GEMM+bias 对 tuned `bmm_fp8`+add 的比值为中位 0.70x、最差 0.90x；8192 以下在 96 个点中 4 点落后、最多 7%（即"up to 7%"指小 M 处不切 epilogue 的理由）。

## 验证与回归风险

新增两组测试：门控测试验证 min_rows−1 走 bmm_fp8、min_rows 走 epilogue（两路输出一致，仅能由门控错误暴露）；CUDA 一致性测试用 rtol 1.6e-2 / atol 0.5 对齐 epilogue 与 bias-free+add 参考（epilogue 在舍入前加 bias，量级 ~20 的输出可差一个舍入步）。精度：Ko-StrategyQA / MIRACL-ko 各 60 题 Recall@10 0.889/0.934 与 main 持平，mean |logit−FP32| 0.142/0.191 vs 0.143/0.191。风险面小：仅影响 SM90+ 上走 bmm_fp8 的带 bias per-tensor FP8 linear，且只在 ≥8192 行切换后端；小 M、其他量化方案不变。

## 关联

前序 #33148（per-tensor FP8 linear 改走 FlashInfer bmm_fp8）；外部依赖 flashinfer-ai/flashinfer#8811（bmm_fp8 加 bias，in review）。无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
