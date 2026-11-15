# PR #59384: [Kernel] Fuse native BF16 ReLU2 with NVFP4 quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @samnordmann |
| 合并 | 2026-11-12 |
| 分类 | Quantization(tier: major) |
| Label | performance, ready, torch.compile, nvidia, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/59384 |

## 一句话结论

在 SM10x 上把 Nemotron shared experts 的 native BF16 ReLU2 与 NVFP4 量化融合为一个 Triton kernel:消除一次 launch 与 BF16 intermediate;2K/4K tokens 下 activation+quantization 延迟降 24–31%(含 down GEMM 降 13–20%)。

## 动机

Nemotron shared experts 路径上 ReLU2 与 NVFP4 量化分两个 kernel,中间还要物化一份 BF16 activation。本 PR 在 SM10x 上把两者融合(复用 #59612 的 `QuantizedActivation` consumer),不改 GEMM。

## 优化点

1. 新增 `relu2_nvfp4_quant.py`:Triton kernel 一次完成 BF16 ReLU2 + block16 NVFP4 量化(128×4 swizzled scales);输出与 `scaled_fp4_quant` bitwise 一致,可喂给任意 `kNvfp4Dynamic` consumer(FlashInfer CuTe-DSL 与 CUTLASS 均验证 bit-exact)。
2. 每个 K 只有一个 Triton specialization(kernel 无 M 相关 constexpr,M 是 `do_not_specialize` 参数),profile run 即完成编译,无需 startup warmup 注册。
3. 经 `direct_register_custom_op` 注册为 opaque custom op,把依赖 M 的 launch geometry 挡在 Dynamo 之外——guard-free Dynamo 下 dynamic rows 仍安全(内核里有 `assert m <= _relu_squared_nvfp4_max_rows(k)` 的 int32 索引保护)。
4. 门控 `_relu_squared_nvfp4_quant_supported`:只融合 native 语义的 ReLU2(CUDA ReLU2 会抑制 NaN,不能与 native producer 混用,`_uses_native_relu_squared` 检查 `forward_native`);x 须为 bf16、2D、contiguous、`K % 16 == 0`、行数不超过 int32 安全上限;scale 须为单元素 fp32 tensor。不满足时保持既有 activation/LoRA fallback。
5. 无 scheduler、runner、通信或依赖改动;与 routed-MoE(#56740)、GELU(#59467)、normalization(#61073)的融合互不重叠;不声明 DISAGG、TP>1 或 default-stack serving 收益。

## 关键改动

```text
vllm/model_executor/layers/fusion/relu2_nvfp4_quant.py(新增): _relu_squared_nvfp4_quant_kernel + custom op relu_squared_nvfp4_quant + int32 行数上限
vllm/model_executor/layers/fusion/fused_act_quant.py: (ReLUSquaredActivation, kNvfp4Dynamic) 注册(SM10x)+ _relu_squared_nvfp4_quant_supported 门控
tests/kernels/test_fused_quant_activation.py: bitwise 对照 / NaN·Inf·饱和 / fallback / 单特化(不重编译)测试
```

```python
        # ReLU propagates NaN; tl.maximum otherwise defaults to maxnum semantics.
        lo = tl.maximum(lo, 0.0, propagate_nan=tl.PropagateNan.ALL)
        hi = tl.maximum(hi, 0.0, propagate_nan=tl.PropagateNan.ALL)
        # Preserve the materialized BF16 ReLU2 boundary before computing amax.
        lo = (lo * lo).to(tl.bfloat16).to(tl.float32)
        hi = (hi * hi).to(tl.bfloat16).to(tl.float32)
        # Match __hmax2 / __hmax in the CUDA quantizer: ignore individual NaNs.
        amax = tl.reduce(_max_ignore_nan(lo, hi), 1, _max_ignore_nan)
```

## 性能声明

- **Kernel microbenchmark(revision `462744b0`,GB200,Nemotron shared-expert 尺寸 K=5376 / N=4096,普通 native compiler 设置,warm CUDA graphs,每 shape 8 组配对样本)**:

| Tokens | ReLU2 + quantization (µs) | Including down GEMM (µs) | Component speedup |
|---:|---:|---:|---:|
| 1 | 2.56 → 1.54 | 6.53 → 6.28 | 1.04× |
| 128 | 3.33 → 2.31 | 8.16 → 7.77 | 1.05× |
| 2,048 | 17.17 → 12.98 | 35.94 → 31.24 | **1.15×** |
| 4,096 | 35.60 → 24.46 | 74.63 → 59.85 | **1.25×** |

  所有配对均改善;配对 component speedup 在 2K 为 1.147–1.151×、4K 为 1.246–1.258×。Nsight 确认 **3 → 2 kernels**,GEMM kernel 与 grid 不变。
- **正确性**:325 项 GPU checks 与 pre-commit/mypy 通过;full-model oracle 707,680 次 fused 调用零 value/scale/padding 错误、启动后无编译;GSM8K 当前 instrumented run 90.75%,此前受控 A/B 四次 run 全部 91.66% 且 tokens 完全一致(非新栈配对比较)。
- **Revision `60a2df16`(移除 warmup),B300**:与 `462744b0` bitwise 一致,kernel 延迟再降 0–13%;Nemotron-3-Super-120B-A12B-NVFP4 上 GSM8K(1319 questions,5-shot,TP1,CUDA graphs)93.6% vs main 93.7%。
- **Serving**:prior-stack C50、2K 输入/1K 输出 median throughput 2359 → 2375 output tokens/s(+0.66%,六 trials/arm,区间重叠)——正文明确"不是 robust 的 serving speedup claim"。
- **Prefill-heavy serving(同 revision)**:AGG TP1,C50,8K 输入/128 输出;fresh-cache B–C–C–B,每实现/预算六次 100-request trials;两次完整重试均完成:24/24 trials、2400/2400 requests 通过。

| Token budget | Baseline output tokens/s, median [range] | Fused output tokens/s, median [range] | Median change |
|---:|---:|---:|---:|
| 2,048 | 202.87 [198.90–204.09] | 204.56 [201.90–208.36] | +0.83% |
| 4,096 | 386.05 [379.85–394.53] | 392.02 [389.71–397.70] | +1.55% |

## 验证与回归风险

- 测试覆盖:与 unfused 路径 bitwise 相等(data 与 swizzled scales 均 `rtol=0, atol=0`)、padding scale 必须清零、NaN/Inf/饱和输入在 4×4 scale 组合下与 `scaled_fp4_quant` 逐位一致、三种 activation 模式(native/compiled_native/cuda)下 dispatch 语义正确、超限输入经 FakeTensorMode 触发 fallback(不分配大内存)、每 K 单特化(换 M 不重编译,`jit_post_compile_hook` 断言)。
- `skipif` 限定:仅在 SM10x(device capability family 100)验证。
- 回归风险:仅 native 语义 ReLU2 参与融合(CUDA ReLU2 抑制 NaN);int32 索引上限保护超大 M;serving 层面收益较小(+0.66%~+1.55%)且作者自认 serving claim 不 robust。

## 关联

- #59612(复用其 `QuantizedActivation` consumer)。
- 相邻独立融合:#56740(routed-MoE)、#59467(GELU)、#61073(normalization)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
