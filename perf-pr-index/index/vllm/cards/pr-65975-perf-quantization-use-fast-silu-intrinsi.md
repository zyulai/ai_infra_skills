# PR #65975: [Perf][Quantization] Use fast SiLU intrinsics in Humming MoE activations

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-11-10 |
| 分类 | MoE (tier: major) |
| Label | quantization |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数为 1 个文件(diff 完整,未见截断) |
| 链接 | https://github.com/vllm-project/vllm/pull/65975 |

## 一句话结论

Humming MoE w2 activation+quant 融合 kernel 的 SiLU 系表达式改用 `__fdividef` + FTZ exp2 快速内联:B300 prefill 尺寸快 16–30%,DeepSeek-V4.1-Flash 的 clamped SwiGLU 最高约 40%;量化输出 MXFP4/MXFP8 逐位一致。

## 动机

vLLM 传给 Humming `process_input` 的激活表达式(MoE w2 activation 与输入量化融合的 kernel)目前写作 `a / (1.f + expf(-a))`。Humming 编译该 kernel 时关闭 fast math(`ProcessInputKernel.disable_fast_math = True`,vllm-project/humming#80 特意为之,目的是保护量化/Hadamard/scale 计算的精度),于是 SiLU 被编译成精确 `expf` 加 IEEE 除法。ncu 显示在 prefill 尺寸下该 kernel 是 instruction-issue bound(IPC ~3.5/4,issue slots 80–87% busy),SiLU 数学主导指令数——这是可以局部消除的瓶颈。

## 优化点

只改激活表达式字符串,量化、Hadamard 与 scale 数学保持精确。两层:(1) `__fdividef` 替代 IEEE 除法——标准 CUDA intrinsic;(2) 内联 PTX `ex2.approx.ftz.f32`(先乘 log2 e)替代 `__expf`,后者在无 fast math 下每元素带 denormal guard(一条 `FSETP` 加两条 predicated `FMUL`)。FTZ 在此处是精确的:`e^x` 只在 SiLU(a) == a 处下溢。`ex2.approx.ftz.f32` 是所有目标上都有的单条 PTX 指令,非 arch-specific。不整体开 `--use_fast_math`,因为它会同时给量化/scale 代码引入 FTZ、近似除法与近似 sqrt(humming#80 特意关掉),把 intrinsics 限定在激活字符串里使改动局部化。覆盖 SILU / SILU_NO_MUL、clamped SiLU(SWIGLUOAI、SWIGLUOAI_UNINTERLEAVE)、SITU、SWIGLUSTEP;GELU(`erff`)与 `tanhf` 不动。边界行为:极负输入时 `1 + exp(-a)` 上溢、`__fdividef` 返回 0 正是 SiLU 极限;大正输入返回 a。decode 尺寸(N=2048、512 行)kernel ~5.5µs,launch-latency bound,无变化。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/humming/activation.py: 
  _SILU_IMPL / _CLAMPED_SILU_IMPL / SITU / SWIGLUSTEP 表达式注入快速 exp 与 __fdividef
```

```python
_EXP_FTZ = (
    "[](float x) { float y; "
    'asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f)); '
    "return y; }"
)
_SILU_IMPL = f"__fdividef(a, 1.f + {_EXP_FTZ}(-a))"
# _CLAMPED_SILU_IMPL / SITU / SWIGLUSTEP 同样用 {exp} 注入 _EXP_FTZ
```

## 性能声明

ncu(`--clock-control none`,B300,SiLU-and-mul bf16 输入 rows×2N):bf16 out N=2048/65536 行 137.6→115.4 µs(vLLM 参考核 `silu_and_mul` 132.0 µs);fp8 block-128 156.5→118.4 µs(参考 682 µs);mxfp4 N=2048 153.2→115.5 µs(参考 171.3)、N=768 63.7→48.6 µs;即 prefill 尺寸 16–30%。FTZ 单独(第二 commit):量化输出 prefill 尺寸再 +4–10%(MXFP4 N=2048/131072 行 232.6→215.0 µs)。DeepSeek-V4.1-Flash w2 形状(clamped SwiGLU limit 10,block-FP8,grouped-EP,B300):N=2304(EP)8192 tok 180.2→126.7 µs、16384 tok 355.7→248.6 µs;N=640(TP4)105.5→78.95 µs。H200(SM90)同 harness:各形状 +15–41%(最高 +53%,N=1152 FP8 per-token),如 N=2304 EP 16384 tok 386.2→289.6 µs;无激活的 w13 量化不变。精度:65,280 个有限 bf16 gate 值全扫 vs fp64 参考,fp32 输出最大相对误差 3.3e-6(精确版 1.8e-7),bf16 输出仅 3/65,280 值不同;8192×2048 输入下 MXFP4/MXFP8 逐位一致,FP8 per-token/block-128 约 16.7M 字节中 ~10 字节不同。端到端 gpt-oss-20b(TP1,B300):prefill 吞吐 +1.9%(MXFP8 act)/+1.7%(FP8 per-token),n=3、离散 ≤0.3%;GPQA 0.569→0.579、GSM8K 0.299→0.292。

## 验证与回归风险

`pytest tests/kernels/core/test_activation.py tests/kernels/moe/test_moe.py -k humming` 175 passed;既有的 `test_humming_activation_matches_framework`(CI 内)对每个 Humming 激活按 fp16/bf16/fp32 与 vLLM 参考比对(含边缘与 clamp 值)。GSM8K(1319 题,5-shot,既有 Humming 配置,B300 双臂并跑):`Qwen3-30B-A3B-FP8-block-humming` 0.850→0.876、`gpt-oss-20b-humming-act-fp8` 0.307→0.315,差异在 run-to-run 噪声内。风险面小:改动仅为激活表达式文本;FP8 路径量化字节有 ~1e-6 级差异(作者已注明 dequantize 后误差与精确 SiLU·up 一致到 7 位有效数字);覆盖面依赖 Humming 上游 `disable_fast_math` 行为保持。作者另列 follow-up(不在本 PR):Humming `process_input` 启动配置启发式在大 grid 上低估向量宽度,上游修复可再添 ~1.4–1.9% 端到端、kernel 级 10–55%。

## 关联

同区域:#59514(最接近的 open PR,改的是 vLLM 自有 `silu_and_mul_quant` CUDA kernel,非 Humming 路径)。上游:vllm-project/humming#80(`disable_fast_math` 的由来,本 PR 在其约束下局部化提速);启动配置启发式修复归属 vllm-project/humming(后续)。pack 内无其他同分类 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
