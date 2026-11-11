# PR #48442: [Perf] diffusion: Accelerate exact SANA-Video RoPE on SM103

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-11-08 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(正文注明:两个文件 +95 行,即 37 行 kernel + 58 行回归测试;diff 摘录完整) |
| 链接 | https://github.com/sgl-project/sglang/pull/48442 |

## 一句话结论

SANA-Video 交错 RoPE 在 SM103 上改用定向 FP32 区间运算证明 FP64 结果,B300 上 2,000 次 RoPE 调用 2835→818 ms(−71%),saved-video E2E −8.11-8.19%。

## 动机

SANA-Video 的 interleaved RoPE 在 B300 上每个 native 50-step 请求要花 2.835 秒、2,000 次 kernel 调用。现有实现按"分别舍入的 FP64 乘积与求和,再经 FP32 转成 BF16"的两段舍入契约输出;本 PR 要在完整保持该输出契约的前提下把这条路径加速。

## 优化点

SM103(`__CUDA_ARCH__ == 1030` guard,其他架构保持原代码)上,`rotate_pair` 先用定向 FP32 区间运算包围原 FP64 表达式:对 FP64 的 cos/sin 分别取向下/向上舍入的 FP32 端点,各乘积与加减按操作数符号选择 `__fmul_rd/ru` 等方向,得到每个 BF16 输出的区间 [lo, hi]。当上下界转换成 BF16 后端点位完全相等且有限时,即可证明原 FP64 两段舍入结果必然落在该 BF16 值上,直接输出;歧义边界与非有限值回落到既有 FP64 运算;带符号零的对保留各自的符号规则。indexing、layout、launch 与公开 API 不变,无新 runtime flag;改动为两个文件 +95 行(37 行 kernel、58 行回归测试)。

## 关键改动

```text
python/sglang/kernels/jit/csrc/diffusion/interleaved_rope_fp64.cuh: rotate_pair 增加 SM103 定向 FP32 区间 fast path,未证明时回落 FP64
test/registered/kernels/ops/diffusion/test_interleaved_rope_fp64.py: 特殊值/消去与带符号零/BF16 tie 舍入三组 bitwise 测试
```

```cpp
const float cl = __double2float_rd(cos), ch = __double2float_ru(cos);
const float sl = __double2float_rd(sin), sh = __double2float_ru(sin);
...
const T lo0 = static_cast<T>(__fsub_rd(acl, bsh));
const T hi0 = static_cast<T>(__fsub_ru(ach, bsl));
const auto bits0 = __bfloat16_as_ushort(lo0), bits1 = __bfloat16_as_ushort(lo1);
if (bits0 == __bfloat16_as_ushort(hi0) && bits1 == __bfloat16_as_ushort(hi1) && ...
```

## 性能声明

B300 SXM6 AC(SM103,275040 MiB,driver 590.48.01),native SGLang backend,832×480×81 帧、16 FPS、50 steps、CFG6、CPU seed42,BF16 DiT/text encoder 与 native FP32 solver/noise/VAE。Eager(18 engines,含 A/A 与回文配对):saved-video E2E main 24.476-24.481 s → final 22.471-22.493 s,reduction 8.114-8.193%(short/long × 两轮);cleanup 变化 −0.0462% 至 +0.0198%。Profile 诊断 trace:2,000 次 RoPE 调用 2835.259→818.106 ms(−71.145%);GPU event time 总和 24,435.554→22,417.468 ms,RoPE 节省 2,017.154 ms 基本解释全部差异。BCG 五臂(30 engines):Main→Final BCG reduction 8.484-8.515%;final BCG 较 final eager 再省 5.453-5.522%;final BCG 对 Omni 的 gate FAIL(Omni full+cache 19.03-19.11 s vs final BCG 21.24-21.26 s,约 0.895-0.900x,即 BCG 相对 Omni 慢约 11.2-11.7%)。NCU 计数器不可用(ERR_NVGPUCTRPERM),无硬件计数器瓶颈结论。

## 验证与回归风险

9 个 focused RoPE GPU tests 通过:全部 BF16 编码(含 subnormal/带符号零/非有限)、极端系数、cancellation 与 signed zero、两段 BF16 tie 舍入;NaN mask 一致,所有非 NaN 位精确。8 个既有 SANA 单测通过。正确性规模:108 个 eager formal 视频(含 warmup)与冻结 main 在 MP4 SHA256 与解码 RGB 逐字节一致;单 GPU 兼容套件 60 个视频(mixed eager/BCG、lossless/exact/high、CFG1、TP2/CFG2 的 85-frame 契约)全部精确;7 种源架构 compile 检查(SM90a/SM100a main/final 指令流一致,SM103a cleanup 前后一致;作者注明这些是 compile 检查而非那两块卡的运行时测试);BCG 混合引擎 650 次 serving replay 无 recapture、500 次签名 miss 均来自质量/形状/序列长度变化。风险:fast path 仅限 SM103,其他架构无收益(仍走 FP64);区间证明只覆盖有限值与 signed-zero 规则;相对 Omni 的延迟门未通过(PR 声明无新增 BCG 场景);eager 峰值显存 22306 MiB 不变,BCG 档约 25.7 GB。

## 关联

无(素材正文未引用其他 PR 编号;实验在 commit `77c3bcbe...`(main 基线)/`f16163d1...`(final head)上进行)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
