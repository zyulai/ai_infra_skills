# PR #40194: [Attention] TurboQuant: remove redundant random signs, add prior art attribution

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @dalistarh |
| 合并 | 2026-04-18 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, v1, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/40194 |

## 一句话结论

TurboQuant 去掉逐层随机符号翻转，Llama-3.2-3B 4bit PPL 7.074（+1.83%），无回归。

## 动机

PR 正文指出 TurboQuant 的 Hadamard 旋转此前为每层附加随机符号翻转（per-layer random sign flips）。作者认为这些符号对 Lloyd-Max 量化质量没有作用：一是量化器关于零对称，二是随机性反而会引入方差。因此逐层随机符号、`_tq_signs` buffer、`generate_wht_signs()`、逐层 seed 逻辑以及 `TurboQuantConfig` 的 `seed` 字段都属于可去除的冗余。正文同时补充历史归属：Hadamard 旋转 + 优化标量网格 + 可选重归一化是 HIGGS（Malinovskii et al., NAACL 2025; arXiv:2411.17525）的标量情形，并首次用于 KV-cache 压缩见于 "Cache Me If You Must"（Shutova et al., ICML 2025; arXiv:2501.19392），二者均早于 TurboQuant 论文；TQ 论文的 JL 改进可能改善理论保证，但原 PR 未发现实际影响。

## 优化点

所有层改为共享同一个纯 Hadamard 矩阵（pure Hadamard），删除 `_tq_signs` buffer、`generate_wht_signs()` 与逐层 seed 逻辑，并从 `TurboQuantConfig` 移除 `seed` 字段；`attention.py` 的 `_init_turboquant_buffers` 不再注册符号 buffer，`turboquant_attn.py` 里 `_tq_PiT` / `_tq_Pi` 直接取 H（因 Sylvester Hadamard 对称，H = H^T）；`__init__.py` 文档由 "random rotation + per-coordinate Lloyd-Max scalar quantization" 改为 "Hadamard rotation + per-coordinate Lloyd-Max scalar quantization"；测试从 WHT rotation 相关（orthonormal、self-inverse、signs 确定性）改为 Hadamard orthonormality 与 symmetry 测试。

## 关键改动

```text
vllm/model_executor/layers/attention/attention.py: 删除 _tq_signs buffer、generate_wht_signs 导入与逐层 seed 计算，仅保留 centroids
vllm/model_executor/layers/quantization/turboquant/config.py: 移除 seed 字段
vllm/model_executor/layers/quantization/turboquant/quantizer.py: 移除 generate_wht_signs
vllm/v1/attention/backends/turboquant_attn.py: _tq_PiT/_tq_Pi 直接使用纯 Hadamard 矩阵
tests/quantization/test_turboquant.py: 删除 sign 相关测试，新增 Hadamard orthonormality/symmetry 测试
```

```diff
-"""TurboQuant: Near-optimal KV-cache quantization for vLLM.
+"""TurboQuant: KV-cache quantization for vLLM.
 
-PolarQuant compression: random rotation + per-coordinate Lloyd-Max
-scalar quantization for keys, uniform quantization for values.
+Hadamard rotation + per-coordinate Lloyd-Max scalar quantization for
+keys, uniform quantization for values.
```

## 性能声明

PR 正文未给出吞吐/延迟类定量性能数据，仅给出量化质量验证。Empirical verification（synthetic）：5 种源分布（Gaussian、sparse、heavy-tailed、correlated、uniform）、3 种 bit widths（2/3/4-bit）、5 个 random seeds 下，Angular error、relative MSE 与 inner product correlation 在 pure 与 randomized Hadamard 之间 identical to 6+ decimal places。Wikitext-2 PPL（模型 Llama-3.2-3B）：Baseline (FP16 KV) 6.947；TQ 4bit+NC pure Hadamard 7.074（Δ baseline +1.83%）；TQ 4bit+NC randomized Hadamard 7.082（+1.94%）；TQ 3bit+NC pure Hadamard 7.376（+6.17%）；TQ 3bit+NC randomized Hadamard 7.368（+6.06%）。正文结论：pure 与 randomized 在噪声范围内（~0.1% PPL），so no regression。硬件与推理 workload 条件 PR 未注明测试条件。

## 验证与回归风险

Test plan 列出：Existing TurboQuant unit tests pass（已更新为删除 sign 相关测试、新增 Hadamard orthonormality 与 symmetry 测试）；Store→decode round-trip test passes with pure Hadamard；Verified no remaining references to `_tq_signs` 或 `generate_wht_signs` in codebase。回归风险：改动触及所有使用 TurboQuant KV-cache 量化的层（旋转矩阵由逐层随机变为共享纯 Hadamard），可能影响 2/3/4-bit 各档量化精度；正文以 PPL 与合成分布验证说明无回归，PR 未注明其它受影响路径或硬件。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
