# PR #35120: [FlashInfer v0.6.18] add FlashInfer CuTe DSL NVFP4 W4A16 mode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zianglih |
| 合并 | 2026-09-01 |
| 分类 | Quantization(tier: major) |
| Label | documentation, quant, deepseek, blackwell, run-ci |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及约 12 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/35120 |

## 一句话结论

FlashInfer CuTe DSL NVFP4 新增 opt-in W4A16 模式(权重 NVFP4、激活/输出 BF16),MoE 与 dense linear 均可启用,精度/吞吐验证齐备。

## 动机

为 FlashInfer CuTe DSL NVFP4 增加权重保持 NVFP4、激活与输出留在 BF16 的 W4A16 计算模式,避免激活量化的额外误差与开销。依赖 FlashInfer v0.6.18 final release(`69ff11f`,含 CuTe DSL NVFP4 MoE W4A16、SiTU、workspace-lifetime 与 SM100/SM103 dense W4A16 API);SGLang main 已通过 #36954 pin 0.6.18,本集成故意不为更早的 FlashInfer API 提供兼容 fallback。注册测试覆盖 online 量化 Nemotron MoE 精度、serialized ModelOpt Qwen3 dense+MoE 的 disk-reload 不变性,以及既有 dense W4A4 数值路径。

## 优化点

新增环境变量 `SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16`(默认 false)。MoE 路径:`CuteDslMoEWrapper(quant_mode="w4a16")`,直接传原始 BF16 激活、省略 GEMM1/GEMM2 activation scales;复用既有 packed NVFP4 weights 与 CuTe MMA-layout block scales,但为 online 量化与 serialized ModelOpt 权重构造 weight-only 的 decode alphas(静态 checkpoint activation scale 置 1,与 per-token 动态量化同理)。Dense 路径:serialized ModelOpt NVFP4 保留在 `ModelOptFp4LinearMethod`;当该 env 与 `--fp4-gemm-backend flashinfer_cutedsl` 同时选中时,注册的 `fp4_gemm` compiler boundary dispatch 到 FlashInfer `mm_bf16_fp4`,否则既有 W4A4 `mm_fp4` 路径不变;权重经 `prepare_bf16_fp4_weights` 准备,保留 loader 拥有的 source scales 供 disk reload,并维持稳定的 prepared scale/alpha 绑定以兼容 decode CUDA graphs。支持 standard CuTe DSL v2 路径(无 A2A 或 FlashInfer A2A,SM100/SM103);FlashInfer A2A 保持 BF16 dispatch(抑制自动 NVFP4 dispatch、拒绝显式配置),CuTe DSL v1 DeepEP masked 路径配 W4A16 直接 raise。DeepSeek 的 W4A4-only GEMM+SwiGLU fusion 与 Qwen3.5 silu fp4 fusion 对 W4A16 保持关闭(守卫加 `quant_mode == "w4a4"`)。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py: quant_mode="w4a16" 选择与 BF16 激活直通
python/sglang/srt/layers/quantization/modelopt_quant.py: fp4_gemm 增 w4a16 分支(mm_bf16_fp4)与 dense W4A16 权重准备
python/sglang/srt/environ.py: SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16(默认 false)
python/sglang/srt/arg_groups/moe_hook.py: W4A16 要求 BF16 FlashInfer MoE dispatch
test/registered/backends 与 rl 目录: online MoE 与 disk-reload 的 W4A16 用例
```

```python
if quant_mode == "w4a4":
    return flashinfer_fp4_gemm(
        input, weight, input_sf, weight_sf, alpha, out_dtype, backend=backend
    )
elif quant_mode == "w4a16":
    from flashinfer import mm_bf16_fp4

    return mm_bf16_fp4(
        input,
        weight,
        weight_sf,
        alpha,
        backend=backend,
        out_dtype=out_dtype,
    )
```

## 性能声明

作者注明"No serving-performance benchmark was rerun",以下速度数据为历史记录。A2A 对照(historical head `508aef613`,MoE W4A16 + dense W4A4,`nvidia/Qwen3-30B-A3B-NVFP4`,TP4/DP4/DP attention,128 并发、8192 in/2048 out):output tok/s 8819.89(no A2A)→10126.28(FlashInfer A2A),total token throughput 44099.43→50631.38;作者注明该表表征 A2A 选择而非最终 dense+MoE W4A16 组合的性能。accuracy 测试中的吞吐观察(8x B300 SM103,4 GPU,带 EAGLE):online NVFP4 MoE 文件(main-merge 后回归)CuTe DSL W4A4 score 0.975、1530.191 token/s、avg_spec_accept_length 3.105;W4A16 score 0.965、1858.117 token/s、accept 3.130。早期 release retest:W4A4 0.970/1568.630,W4A16 0.980/1662.258。

## 验证与回归风险

新增 nightly `4-gpu-b200` 端到端用例(Nemotron-3-Super-120B-A12B-FP8 online NVFP4 量化,FlashInfer A2A)与 TP4/DP4/EP4 serialized Qwen3-30B-A3B-NVFP4 disk-reload 用例(两次 release/reload/resume 必须保持确定性 decode text 与 token ID 完全一致,logprob 差 1e-4 内);lm-head guard 单测与 dispatch fixture 修正(声明 `quant_mode="w4a4"`)。回归面:W4A16 默认关闭;v1 DeepEP masked 路径显式拒绝;`SGLANG_MOE_NVFP4_DISPATCH` 与 W4A16 组合被拒绝。作者详述合并后 CI 失败为测试 fixture 漂移(production 提供 quant_mode 而 fixture 未声明),以 follow-up commit 修复,无运行时代码变更。pack 未包含 CI States 记录。

## 关联

依赖:flashinfer v0.6.18 final release;SGLang main 经 #36954 pin 0.6.18;上游 flashinfer#4486(测试 skip 块相关)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
