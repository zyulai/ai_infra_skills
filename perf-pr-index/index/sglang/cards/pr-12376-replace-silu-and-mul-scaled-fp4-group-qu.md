# PR #12376: Replace [silu_and_mul_]scaled_fp4_group_quant by Flashinfer equivalent

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wenscarl |
| 合并 | 2025-11-13 |
| 分类 | Quantization |
| Label | documentation, performance, quant, sgl-kernel, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/12376 |

## 一句话结论

把 nvfp4 grouped quant 算子换成 Flashinfer 等价 API,统一维护路径,端到端吞吐基本持平。

## 动机

Flashinfer #1927 引入了两个 nvfp4 量化 API:`silu_and_mul_scaled_nvfp4_experts_quantize` 与 `scaled_nvfp4_grouped_quantize`,实现上与 sgl-kernel 的 `silu_and_mul_scaled_fp4_grouped_quant`、`scaled_fp4_grouped_quant` 等价。本 PR 改用 Flashinfer 版本以便后续维护,消除 sgl-kernel 侧的重复实现。

## 优化点

这是算子来源替换而非新内核。核心是 `flashinfer_cutedsl_moe.py` 的 masked MoE 路径:up-GEMM 前的 fp4 grouped 量化改调 `scaled_fp4_grouped_quantize`,down-GEMM 前的 silu+量化融合改调 `silu_and_mul_scaled_nvfp4_experts_quantize`。两套 API 参数顺序不同(masks 与 scales 位置互换)。kernel benchmark 脚本与各处单测的 import、调用同步更新;融合与非融合路径的输出一致性由 `torch.testing.assert_close` 断言保持。

## 关键改动

```text
python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py: cutedsl masked MoE 的 a1/a2 fp4 grouped quant 改调 Flashinfer
benchmark/kernels/quantization/bench_fp4_quant.py: kernel benchmark 切换到 Flashinfer API
sgl-kernel/tests/test_fp4_quantize.py: 分组量化与 silu+mul 融合量化测试改用 Flashinfer
test/srt/test_cutedsl_moe.py、test/srt/test_fp4_moe.py: 端到端 MoE 测试同步替换
```

## 性能声明

PR 附两组对比(PR 未注明测试硬件):

1. GSM8K(256 题,parallel 32,8-shot);server 用 modelopt_fp4 量化 + deepep low_latency + flashinfer_cutedsl MoE runner,TP4/EP4/DP4(模型路径以 $model_str 占位,pack 未注明具体模型)。本 PR:accuracy 0.984、latency 92.829 s、output throughput 312.521 token/s;替换前:0.980、93.843 s、291.104 token/s。
2. bench_serving:nvidia/DeepSeek-R1-0528-FP4,random 1024/1024,并发 256,512 prompts。本 PR:total token throughput 8353.85 tok/s、mean TTFT 14263.65 ms;替换前:8396.60 tok/s、13982.02 ms(本组指标略低于替换前)。

两组方向相反,作者的定位是等价替换而非性能优化。

## 验证与回归风险

依赖 flashinfer 提供 #1927 引入的两个 API,存在版本耦合;参数顺序 masks/scales 互换,新调用方易写错。回归路径集中在 nvfp4/cutedsl MoE(fp4 grouped quant、silu+mul 融合量化、cutedsl masked MoE、fp4 MoE 测试)。GSM8K 精度 0.980→0.984 基本持平。其他仍使用 sgl_kernel 量化算子的 backend 不受影响。

## 关联

上游依赖 flashinfer#1927(正文提及);同库前序/后续 PR 正文未列。无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
