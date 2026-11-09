# PR #63915: [ROCm][Perf] Fuse MXFP8 activation quant into the 1x32 dot_scaled linear for M <= 16

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @RuibinCheung |
| 合并 | 2026-10-29 |
| 分类 | Quantization(major) |
| Label | rocm, ready, quantization, verified |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 内容可用(kernel 与测试两个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/63915 |

## 一句话结论

gfx950 的 MXFP8 1x32 dot_scaled 线性层在 M<=16 的 bf16 小 GEMM 上把激活量化融合进 GEMM kernel,每次调用少一个 launch,MI355X microbench 提速 1.14x-1.64x。

## 动机

`RocmDotScaledMxfp8LinearKernel`(服务 MiniMax-M3/M3.1、ModelOpt/INC MXFP8 的 1x32-scale MXFP8 dense 层)在每个 `tl.dot_scaled` GEMM 前都要单独跑一次 `mxfp8_e4m3_quantize` launch。decode 的小 batch(M=1-16)下,独立 quant launch 及 fp8 激活与其 scale 的往返开销占比高。

## 优化点

`_mxfp8_linear_kernel` 增加 `QUANT_X`/`QUANT_ROWS` 编译期分支:命中时直接加载 bf16 激活,每个 N-tile 在 kernel 内用 `_quant_mxfp8_tile` 做位精确的 MXFP8 量化(E8M0 scale = ceil(log2(amax/FP8_MAX))+127,夹到 [0,254],以 reciprocal 应用),再 `tl.dot_scaled`。关键设计:`BLOCK_M=16` 是最小 MFMA tile,故只量化 `min(16, next_pow2(M))` 行再 broadcast 填满 tile,使量化成本随真实行数而非 tile 大小增长——这是 K=2048 也能盈利的原因;K 改为 `tl.constexpr` 以便 loop unroll/pipeline(否则融合后约 0.8x)。门控 `_select_fused_quant_cfg`:M<=16 且 K<=1024,或 M<=4 且 K=2048;`VLLM_BATCH_INVARIANT` 下不启用(融合与独立路径 tile/K 归约顺序不同);fp16 激活不启用(编译器为整个 kernel 打开 FP16 overflow clamp,输出饱和语义改变)。输出与原路径 bitwise 相等,包括 BLOCK_K/BLOCK_N 不同的 tile。编译变体增多(MiniMax-M3.1 TP4 为 17 个 vs 12),在 eager warmup 完成,实测 graph capture 时间不变。

## 关键改动

```text
vllm/model_executor/kernels/linear/mxfp8/rocm_native.py: kernel 增加 QUANT_X/QUANT_ROWS 融合分支、_quant_mxfp8_tile、_launch、_select_fused_quant_cfg 门控
tests/kernels/quantization/test_rocm_mxfp8_linear.py: 新增 test_rocm_mxfp8_fused_act_quant_bitwise_eq_unfused(5 个 (N,K) × M={1,2,3,5,8,16} 加 K=2048 × M={1..4},spy 验证走融合且 torch.equal)
```

```python
if (
    0 < M <= _FUSED_QUANT_MAX_M
    and x.dtype == torch.bfloat16
    and not envs.VLLM_BATCH_INVARIANT
):
    cfg = _select_fused_quant_cfg(M, N, K)
    if cfg is not None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        _launch(x, None, w, w_scale, out, *cfg)
        return out
```

## 性能声明

Microbenchmark(MI355X,"separate" = `mxfp8_e4m3_quantize` + GEMM):K<=1024、M=1-16 时 1.14x-1.64x;K=2048、M=1-4 时 1.03x-1.27x。示例:6144x384 M=1 为 1.64x;8192x512 M=1-16 为 1.44x-1.51x;6144x1024(M3.1 o_proj TP8)M=1 为 1.52x;6144x2048(M3.1 o_proj TP4)M=1 为 1.27x;M=32 或 K>2048 保持 1.00x(不融合)。端到端(MiniMax-M3.1-Flash-Quark-MXFP4,TP4,MI355X,AgentX 并发 1,600 s/次,两次换序运行):P90 TPOT 1.011x,P90 interactivity 1.011x-1.012x,总吞吐 1.004x/1.020x。GSM8K(lm-eval 5-shot 全集 greedy):baseline 96.51 / 本 PR 96.21 strict-match(stderr 0.51/0.53,一个 stderr 之内)。

## 验证与回归风险

新增位相等测试含全零 block(scale byte 夹 0)与超 fp16 范围的极端激活,并用 monkeypatch spy 确认确实走融合路径。作者声明不改变行为的场景:VLLM_BATCH_INVARIANT、fp16、M>16、K=2048 且 M>4、K>2048。风险点:仅限 gfx950(`dot_scaled` 路径),形状在 MI355X 上 tuning;Triton 编译变体每形状最多 6 个,依赖 warmup 在 graph capture 前完成。

## 关联

#63232(同款 K constexpr 改动,e2e 基线基于它)、#52231(VLLM_BATCH_INVARIANT)。同分类(Quantization)另见 #20841、#35088 等卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
