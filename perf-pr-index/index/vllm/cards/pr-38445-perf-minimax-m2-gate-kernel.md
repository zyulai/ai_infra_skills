# PR #38445: [PERF]MiniMax-M2 gate kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-05-30 |
| 分类 | Attention / Kernel |
| Label | performance, ready, ci/build |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/38445 |

## 一句话结论

为 MiniMax-M2 的 FP32 MoE gate 新增专用 `fp32_router_gemm` kernel(SM90+,M≤32,H=3072,E=256):低并发吞吐 +6~32%,c=2 时 +32%。

## 动机

MiniMax-M2 的 MoE gate 需要 FP32 GEMM:原实现先 `hidden_states.to(torch.float32)` 再走 ReplicatedLinear,低并发时最坏要 launch 3 个 kernel——一次 BF16→FP32 转换加 split-K GEMM 拆出的两个 kernel。参照 `dsv3_router_gemm` 的专用 kernel 思路,把转换与 GEMM 融进一个 launch。

## 优化点

新增 `fp32_router_gemm`(adapted from dsv3_router_gemm_float_out.cu):activation(bf16 或 fp32)× fp32 weight → fp32 输出,固定形状 H=3072、E=256、M≤32,按 M=1..32 × 两种输入类型显式实例化;每 expert 一个 block(128 线程),`VPT=16/sizeof(InputT)` 向量化加载(fp32 一次 float4,bf16 一次 uint4 + 元素转换),warp butterfly 归约 + shared memory 二段归约;SM90+ 上用 PDL(`griddepcontrol.wait`/`launch_dependents` + ProgrammaticStreamSerialization)隐藏 launch 间隔。`GateLinear` 分派升级为四层:DSV3 kernel → fp32 kernel(M≤32)→ cuBLAS bf16×bf16→fp32 → F.linear;M>32 经 `fp32_router_gemm_dispatch` custom op 运行时回退 F.linear(包成 custom op 是因为 torch.compile 集成不支持对 num_tokens 的运行时分支)。`minimax_m2.py` 的 gate 换成 `GateLinear(out_dtype=fp32)`,forward 不再手动 `.to(float32)`。构建上要求 SM90+ 且 CUDA ≥12.0。

## 关键改动

```text
csrc/libtorch_stable/fp32_router_gemm.cu + fp32_router_gemm_entry.cu: kernel 与 op 注册(M=1..32 实例化)
vllm/model_executor/layers/fused_moe/router/gate_linear.py: Tier 2 分派与 dispatch custom op
vllm/model_executor/models/minimax_m2.py: gate 换 GateLinear,去掉显式 FP32 转换
benchmarks/kernels/benchmark_router_gemm.py: 三类 router gemm 统一 benchmark
```

```cpp
for (int ki = 0; ki < k_iterations; ki++) {
  int const k_base = k_bases[ki];
  float b_float[VPT];
  load_weight<VPT>(b_col + k_base, b_float);
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    float a_float[VPT];
    load_activation<InputT, VPT>(mat_a + m_idx * kHiddenDim + k_base, a_float);
    for (int k = 0; k < VPT; k++) acc[m_idx] += a_float[k] * b_float[k];
  }
}
```

## 性能声明

- 条件:4x NVIDIA GB200(189471 MiB),nvidia/MiniMax-M2.5-NVFP4 TP4,`VLLM_FLASHINFER_MOE_BACKEND=latency`,`--performance-mode interactivity`,kv fp8,compilation mode 3;SPEED-Bench throughput_16k/low_entropy(input ≤10240、output 1536 tokens),并发 1–64;main(b2bc736b1) vs feature(fe8426d3c)。
- Output tok/s:c=1 +6.4%(147.56→157.04),c=2 +32.2%(234.32→309.81),c=4 +8.5%,c=8 +7.6%,c=16 +5.9%,c=32 +14.2%(2381.54→2720.66),c=64 −0.2%。
- TPOT:c=2 −25.2%(8.32→6.22 ms),c=32 −13.1%;TTFT:c=2 −29.0%,c=16 +10.0%(+47 ms,PR 自标唯一异常);c=64 全部指标差异 <1%(视为统计噪声)。
- 精度:GSM8K 5-shot flexible 0.9401→0.9318、strict 0.9378→0.9280;AIME25 0.8417→0.8333;GPQA Diamond 0.8182→0.8384(+2.5%);作者结论 no accuracy regression detected。

## 验证与回归风险

新增 `tests/kernels/test_fp32_router_gemm.py`(fp32/bf16 activation 对 reference,num_tokens 1/2/4/8/16/32,SM90+ 以下 skip)、`benchmark_router_gemm.py` 统一覆盖 DSV3/GPT-OSS/fp32 三类 router gemm;E2E 七个并发档两分支均无失败。风险:形状硬编码(H=3072、E=256、M≤32),其他模型或 M>32 一律回退 F.linear,收益仅限低中并发;仅 SM90+ 且 CUDA≥12.0 编译进 stable ext;c=16 的 TTFT +10% 异常 PR 未解释;GSM8K/AIME25 有约 −1% 的分数波动,非严格持平。

## 关联

参照 `dsv3_router_gemm`(GateLinear Tier 1 已有);索引同区域另有 pr-2453(DeepSeekMoE fused MoE kernel)、pr-46340(batched MoE operand loads)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
