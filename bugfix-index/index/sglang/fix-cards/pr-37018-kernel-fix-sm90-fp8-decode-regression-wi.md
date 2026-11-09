# PR #37018: [Kernel] Fix SM90 FP8 decode regression with benchmarked M/K/N routing

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37018](https://github.com/sgl-project/sglang/pull/37018) |
| 作者 | RunFMe |
| 合并日期 | 2026-08-29 |
| labels | run-ci, jit-kernel, run-ci-extra |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#34318 的 SM90 row/column-scaled FP8 选择器仅凭 K/N 就把 decode GEMM 路由到 `torch._scaled_mm`,H100 W8A8 decode 吞吐从约 215 tok/s 跌到约 190 tok/s;把 Torch 路由收紧到实测过的大 prefill 包络修复。

## 问题现象
SM90(H100/H200)上 row/column-scaled FP8 的 decode 吞吐回归。正文引用 #34318 下报告的 H100 W8A8 回归:模型 `Llama-3.1-8B-Instruct-FP8-dynamic`,测试 `test/registered/quant/test_w8a8_quantization.py::TestW8A8Fp8.test_throughput`,batch size 1,吞吐从 #34318 前约 215 tok/s 降到之后约 190 tok/s;受影响的 down projection 形状为 `M=1, K=14336, N=4096`。作者在 H200 与 H100 上直接 profile 该形状:AOT kernel 在 decode M 下明显更快——H200 上 26.50µs vs Torch 51.12µs(AOT 快 1.93×),H100 上 32.19µs vs 53.55µs(1.66×)。

## 根因
正文披露:#34318 引入的 SM90 row/column-scaled FP8 选择器可以仅基于 K/N 把 decode GEMM 路由到 `torch._scaled_mm`(NVJet),而基准测量显示这些 decode 形状上 AOT kernel 更快——路由判据把 decode/小 prefill 也错误地划给了 Torch。

## 修复方案
改 `python/sglang/kernels/ops/gemm/__init__.py` 的 `_prefer_torch_rowwise_fp8`,把 Torch/NVJet 路由约束到实测有效的大 prefill 包络(作者称 m >= 6144 收益不值得):

```python
m >= 8192 and (
    (k >= 4096 and n >= 6144) or
    (k >= 7168 and n >= 5376)
)
```

```diff
-    return (k >= 5376 and n >= 3584) or (k >= 3584 and m >= 8192)
+    return m >= 8192 and ((k >= 4096 and n >= 6144) or (k >= 7168 and n >= 5376))
```

由此保留在 AOT 上的包括:所有 `M < 8192` 的 decode 与小 prefill、被报告的 `K=14336, N=4096` Llama down projection、窄 TP8 projection(`K=5376, N=3584`、`K=3584, N=5376`)以及测量包络之外的形状;Torch 仍用于稳定的大 prefill 收益场景(wide QKV/gate/up 与 broad down projection)。

验证(PyTorch 2.11.0+cu130、CUDA 13.0、sgl_kernel==0.4.4,CUDA-event 计时):11 个 projection family × 12 个 M 值共 132 例的 broad grid 加随机化边界确认,H200 上新谓词选中的 32 例全部偏向 Torch(中位加速 2.50%,0.17–5.97%),H100 上中位 19.48%(1.36–56.53%);被排除的 Llama down 与窄 TP8 family 在大 M 下接近持平或跨轮变号,故刻意留在 AOT。

## 影响与回归风险
- 影响窗口:自 #34318 合并引入该选择器起,至本 PR 合并(2026-08-29);正文未给出版本号。
- 回归面:仅 SM90 row/column-scaled FP8 GEMM 的算子选择(`_prefer_torch_rowwise_fp8` 谓词)。`M >= 8192` 且满足新 K/N 包络的调用仍走 Torch,其余一律回 AOT——与 #34318 之前相比,decode 性能恢复;若存在未测到的、Torch 确实更快但落在包络外的形状,会随之留在 AOT(作者以两轮随机化边界测量降低该风险)。数值路径不变(均为 scaled MM),无精度影响。

## 关联
- PR #34318(引入选择器): https://github.com/sgl-project/sglang/pull/34318
- 回归报告 comment: https://github.com/sgl-project/sglang/pull/34318#issuecomment-5461656173

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
