# PR #22895: [Kernel] Added flashinfer fp8 per-tensor gemms

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @nvjullin |
| 合并 | 2025-08-26 |
| 分类 | Quantization(major) |
| Label | ready, ci/build, v1 |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/22895 |

## 一句话结论

为 FP8 per-tensor GEMM 接入 flashinfer 后端并在 sm100 上设为默认,llama3 TP1 serving TTFT 改善约 13%,TPOT 基本持平。

## 动机

vLLM 的 FP8 W8A8 per-tensor GEMM 此前主要走 cutlass `cutlass_w8a8_scaled_mm`。flashinfer 新增的 FP8 GEMM(经 autotuning)在同等或更好的性能下对部分尺寸更快:小 M 边际更快、大 M 显著更快;prefill(batch 大 M)正是 TTFT 瓶颈,decode TPOT 不受影响。依赖 flashinfer autotuning(#22346)及 flashinfer PR #1479/#1491(均已合并)与新版 flashinfer 发布。

## 优化点

`w8a8_utils.py` 的后端分派从布尔 `cutlass_fp8_supported` 重构为 `preferred_backend` 字符串(rocm/flashinfer/cutlass/torch):`Fp8LinearOp` 初始化时,在 CUDA 且支持 cutlass 的前提下,若 `has_flashinfer()` 且设备 compute capability ≥ 100 则选 flashinfer,否则 cutlass;ROCm 仍走 rocm 后端,其余走 torch。flashinfer 路径经新增 `vllm::bmm_fp8` custom op 调用 `flashinfer.bmm_fp8`,由 `flashinfer_scaled_fp8_mm` 包装(校验 per-tensor scale、fp8e4m3、2D 输入),bias 在包装层外加。per-channel/per-token 等非 per-tensor 组合仍回落 cutlass。`Fp8LinearOp` 的 `cutlass_fp8_supported` 参数改为 `force_fp8_e4m3fnuz`(ROCm e4m3fnuz 强制),pad_output 判定改为仅在 torch 后端时启用。

## 关键改动

```text
vllm/utils/flashinfer.py: 注册 vllm::bmm_fp8 custom op,新增 flashinfer_scaled_fp8_mm 包装
vllm/model_executor/layers/quantization/utils/w8a8_utils.py: flashinfer_w8a8_scaled_mm;dispatch_w8a8_scaled_mm 改按 preferred_backend 分派
vllm/model_executor/layers/quantization/fp8.py: Fp8LinearOp 构造去掉 cutlass_fp8_supported
vllm/model_executor/layers/quantization/ptpc_fp8.py: 改传 force_fp8_e4m3fnuz=True
tests/kernels/quantization/test_flashinfer_scaled_mm.py: 新增数值测试(sm100 门控)
```

后端选择节选:

```python
elif current_platform.is_cuda() and not force_fp8_e4m3fnuz \
        and cutlass_fp8_supported():
    if has_flashinfer() and current_platform.has_device_capability(100):
        self.preferred_backend = "flashinfer"
    else:
        self.preferred_backend = "cutlass"
```

## 性能声明

llama3,ISL=OSL=1024,concurrency=128,max_num_batched_tokens=8192,TP1。旧(cutlass):Mean TTFT 2955.22 ms,Mean TPOT 40.50 ms,输出吞吐 2948.16 tok/s,请求吞吐 2.88 req/s;新(flashinfer):Mean TTFT 2611.05 ms(约 −13%),Median TTFT 1330.74→1182.85 ms,P99 TTFT 10734.06→9319.07 ms,Mean TPOT 39.19 ms(大致持平),输出吞吐 3064.22 tok/s。PR 未注明 GPU 型号(仅从代码推断需 sm100)与量化格式以外的 workload 细节。lm_eval:GSM8K 5-shot flexible-extract 0.944±0.0103,strict-match 0.886±0.0142。

## 验证与回归风险

新增 `tests/kernels/quantization/test_flashinfer_scaled_mm.py`:对比 fp32 参考矩阵乘,含 bias 与 pad 形状、autotune 开/关;仅 compute capability ≥ 100 运行,否则模块级 skip。CI pipeline 加入该测试。风险面:分派逻辑重构触及所有 FP8 linear 路径(ROCm/torch/非 sm100 CUDA 行为应不变,由 compile 相关测试参数化覆盖);flashinfer autotune 缓存与版本依赖;`flashinfer_scaled_fp8_mm` 断言仅支持 per-tensor scale。

## 关联

依赖:#22346(flashinfer autotuning)、flashinfer PR #1479/#1491。同分类为 FP8/低精度 GEMM 后端选型系列;其他编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
