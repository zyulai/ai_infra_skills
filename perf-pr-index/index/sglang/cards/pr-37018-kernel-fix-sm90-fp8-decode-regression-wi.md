# PR #37018: [Kernel] Fix SM90 FP8 decode regression with benchmarked M/K/N routing

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RunFMe |
| 合并 | 2026-08-29 |
| 分类 | Quantization (tier: major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用(单文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/37018 |

## 一句话结论

收紧 SM90 rowwise FP8 GEMM 走 `torch._scaled_mm` 的路由条件,修复 decode 吞吐回归(H100 约 215→190 tok/s)。

## 动机

#34318 引入的 SM90 row/column-scaled FP8 选择器仅凭 K/N 就把 decode GEMM 路由到 `torch._scaled_mm`(NVJet)。已报告的 H100 W8A8 回归:`Llama-3.1-8B-Instruct-FP8-dynamic`,`test_w8a8_quantization.py::TestW8A8Fp8.test_throughput`,batch size 1,吞吐从约 215 tok/s 降到约 190 tok/s,受影响的是 down projection(M=1, K=14336, N=4096)。作者在 H200/H100 上直接剖析该 shape,发现 decode M 下 AOT kernel 明显更快。

## 优化点

只改一处路由谓词 `_prefer_torch_rowwise_fp8`:把 Torch/NVJet 路由约束在实测有稳定收益的大 prefill 包络内——`m >= 8192 and ((k >= 4096 and n >= 6144) or (k >= 7168 and n >= 5376))`,替换原先的 `(k >= 5376 and n >= 3584) or (k >= 3584 and m >= 8192)`。据此保留在 AOT kernel 上的有:所有 M<8192 的 decode 与小 prefill 调用、被报告的 Llama down projection(K=14336, N=4096)、窄 TP8 投影(K=5376, N=3584 与 K=3584, N=5376)以及包络外的 shape;Torch 继续承接稳定有大 prefill 收益的宽 QKV/gate/up 与宽 down 投影。

## 关键改动

```text
python/sglang/kernels/ops/gemm/__init__.py: _prefer_torch_rowwise_fp8 谓词改为基准实测的大 prefill 包络
```

```python
# SM90 benchmarks show repeatable NVJet wins only for large prefill M ...
return m >= 8192 and ((k >= 4096 and n >= 6144) or (k >= 7168 and n >= 5376))
```

## 性能声明

回归 shape 实测(CUDA-event 计时、随机交错采样;PyTorch 2.11.0+cu130,CUDA 13.0,sgl_kernel==0.4.4):M=1, K=14336, N=4096 时 H200 上 AOT 26.50 µs vs Torch 51.12 µs(AOT 1.93x faster),H100 上 32.19 µs vs 53.55 µs(AOT 1.66x faster)。

广域网格(11 投影族 x 12 个 M 值,共 132 例;M=1..16384;百分比是 Torch 相对 AOT 的加速):H200 上新谓词选中的 32 例全部偏向 Torch,中位 2.50%(0.17-5.97%),另两轮随机边界确认 48 例复测全部偏向 Torch;H100 上 32 例中位 19.48%(1.36-56.53%),边界确认 24 例中位 21.02%(2.72-57.38%)。被排除的 Llama down 与窄 TP8 族在大 M 下接近持平或跨轮变号,故刻意留在 AOT。

## 验证与回归风险

diff 未新增测试,依据是上述可复现 benchmark 协议(预热/采样数、exact-workload conditioning、随机化边界确认)。风险面:改动影响 SM90 上所有 rowwise FP8 scaled MM 的 kernel 选择——大 prefill 宽投影继续走 Torch(NVJet),decode/小 prefill 回到 AOT;若硬件/库版本变化使包络失真,需重新基准;对 H100/H200 以外的 SM90 变体未单独测量。CI 状态显示 Extra 与 AMD ROCm 7.2 未通过(AMD 与本改动无交集)。

## 关联

前序 #34318(引入该选择器及其回归报告);正文未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
