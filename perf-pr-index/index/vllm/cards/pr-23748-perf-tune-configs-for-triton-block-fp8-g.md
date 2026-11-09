# PR #23748: [Perf] Tune configs for triton block fp8 gemm H100/H200

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-08-28 |
| 分类 | Quantization(tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材截断(12 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/23748 |

## 一句话结论

用现代 Triton 重调 w8a8 block-FP8 dense GEMM 的 H100/H200 配置,主要改善小 M 批次。

## 动机

vLLM 内置的 triton fp8 block GEMM per-shape 配置是按旧版 Triton 调的,在 Triton 升级后不再是当前硬件上的最优解;小 M(decode/小 batch)场景与 DeepSeek-V3 的特定 shape 尤其吃亏,其中 N=576,K=7168 被点名为关键缺口。

## 优化点

对 H100 与 H200 重跑 tuning,更新/新增一批 `dtype=fp8_w8a8,block_shape=[128,128]` 的 per-shape 配置 JSON:小 M(1-16)的 BLOCK_SIZE_M 普遍从 64 降到 16,GROUP_SIZE_M/num_warps/num_stages 重选;新增 N=12288,K=7168 与 N=24576,K=1536 两套配置文件。同时新增 benchmarks/kernels/bench_block_fp8_gemm.py:覆盖 DeepSeek-V3 全部 10 个权重 shape,以 TFLOP/s 对比 torch-bf16 与 w8a8-block-fp8,是不带 tuning 的复测脚本。

## 关键改动

```text
benchmarks/kernels/bench_block_fp8_gemm.py: 新增 block-FP8 GEMM benchmark(DeepSeek-V3 shapes)
vllm/model_executor/layers/quantization/utils/configs/N=12288,K=7168,...H100/H200....json: 新增配置
vllm/model_executor/layers/quantization/utils/configs/N=24576,K=1536,...H100/H200....json: 新增配置
configs/N=24576,K=7168 / N=32768,K=512 / N=36864,K=7168 (H100/H200): 重调更新(另有 12 个文件在素材 diff 中被截断)
```

```python
DEEPSEEK_V3_SHAPES = [
    (512 + 64, 7168),
    ((128 + 64) * 128, 7168),
    (128 * (128 + 128), 512),
    (7168, 16384),
    ...
]
```

## 性能声明

正文文字仅声明:"Mostly improves performance for smaller M, but crucially gives improvement for N=576,K=7168"。定量结果只以 H100/H200 各三张截图形式给出(素材为图片链接),未提供文字数字,PR 未注明具体加速百分比。

## 验证与回归风险

验证即上述 benchmark 截图;脚本用 do_bench_cudagraph + quantiles [0.5, 0.2, 0.8]。风险面:仅替换 Triton GEMM 的 per-shape tile 配置,不换 kernel,数值结果不变;新配置只覆盖列出的 (N,K) 与设备名,其他设备/shape 沿用原配置;Triton 版本再升级后可能需要重调(这正是本 PR 的起因)。

## 关联

同分类 Quantization 卡:#14068(MoE fp8 block quant tuning 基础设施)、#12978(stream-K for block-quantized GEMM)、#18844/#21476(FP8 量化其他优化)。同属 DeepSeek-V3 shape 覆盖:#19205(MoE 侧配置)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
