# PR #65230: [Quantization] Remove QuTLASS and fp_quant, run QuIP NVFP4 on Humming

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-11-03 |
| 分类 | Quantization(major) |
| Label | documentation, performance, ci/build, nvidia, quantization |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 8 个文件,pack 注明另有 17 个文件因长度省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/65230 |

## 一句话结论

删除 QuTLASS 与弃用的 fp_quant 方法(约 1800 行),QuIP NVFP4(Hadamard 输入变换)改由 Humming kernel 承接,修复 main 上 SM100+ 启动崩溃。

## 动机

main 上 QuIP NVFP4 checkpoint 已无法启动,报错 `'QuantizedActivation' object has no attribute 'shape'`:自 #44260 起 NVFP4 层暴露 `input_quant_key`,producer 传入 QuantizedActivation 而 transform 方法无法处理;SM12x 还因 QuTLASS wheel 只含 sm_100 代码而失败。fp_quant checkpoint 则因 `FPQuantLinearMethod` 未注册 `backward_hadamard_matrix` 加载失败,且该方法自 #31688 起已弃用。这两个回归一直没被发现,因为持有 QuIP perplexity 测试的 CI job 被注释掉了。

## 优化点

路由策略:W4A4、SM100+、`hadamard` 输入变换且 head_dim 16-128、无输出变换、backend 为 `auto`/`humming` 时走 Humming;其余走 generic transform path。Humming 的 block size 16-256 输出与 dense Sylvester 矩阵匹配到 bf16 rounding。同时保证 input transforms 总是接收未量化 activations,顺带修复 generic transform path。删除 `_qutlass_C` 构建依赖、`fp_quant` 方法与约 1800 行代码(含 QuTLASS 的 NVFP4/MXFP4 测试与基准脚本)。

## 关键改动

```text
csrc/qutlass_registration.cpp(删除): QuTLASS torch op 注册
vllm/model_executor/kernels/linear/nvfp4/ 与 compressed_tensors/transform/: Humming/hadamard 路由实现(pack 省略的具体文件,CI 新增对这两条路径的覆盖)
tests/models/quantization/test_nvfp4.py: 新增 quip 与 hadamard_fp4_routing 参数化用例(路由/回退场景)
tests/kernels/quantization/test_nvfp4_qutlass.py、test_mxfp4_qutlass.py(删除): QuTLASS kernel 测试
benchmarks/kernels/benchmark_mxfp4_qutlass.py、benchmark_nvfp4_qutlass.py(删除): QuTLASS 基准
```

## 性能声明

B300(SM103),humming-kernels 0.1.18(FP-Quant 基线数字来自 scratch prototype,QuIP 行在本分支重跑):

- GSM8K(600 题,5-shot,greedy,%):Llama-3.2-1B QuIP NVFP4(float 32.3)QuTLASS 18.8 / Humming A4 17.3-20.5 / Humming A16+Hadamard 22.8;Qwen3-1.7B FP-Quant QAT NVFP4 55.0 / 52.0 / 62.0;FP-Quant RTN NVFP4 23.5 / 21.3 / 45.2;RTN MXFP4 16.3 / 15.3 / 26.7。A4 与 QuTLASS 为统计打平(paired p=0.13-0.93),run-to-run 噪声大(同进程两跑 17.3 与 20.5,600 题中 196 题输出一致)。
- GEMM GPU time(µs,graph replay,真实 down_proj 权重,N=2048,K=6144,warm L2,未手调):M=1:QuTLASS+cutlass 17.9 / QuTLASS+FlashInfer 7.7 / Humming A4 9.7 / A16 9.8;M=1024:36.8 / 24.1 / 15.9 / 41.4;M=8192:164.9 / 146.4 / 75.8 / 240.6。已知损失:对比 CT FlashInfer 路径,M=1-16 约慢 2µs。

## 验证与回归风险

`pytest tests/models/quantization/test_nvfp4.py -k 'quip or hadamard_fp4_routing'` 8 passed;`tests/quantization/test_compressed_tensors.py -k transforms_perplexity` 3 passed;QuIP 生成在 `VLLM_GPU_SYNC_CHECK=error` 下干净;Buildkite kernels job 移除 QuTLASS 测试并新增两条路径覆盖。Breaking:`fp_quant` checkpoint 不再加载(在 main 上本已失败)。未验证:SM120、TP>1(`HadamardTransform` 仍抛 NotImplementedError)。不再提供:QuEST for MXFP4、与 QuTLASS 的 bit-exact 输出(scale rounding 不同)、非 Sylvester dense transform 的 fused 处理。

## 关联

Supersedes #61671、使 #51112 失去意义;根因涉及 #44260、#31688;同批移除型 PR #32688(Marlin 24)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
