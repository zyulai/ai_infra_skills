# PR #38112: [CPU] Added faster exp routine for lower precision data types.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @almayne |
| 合并 | 2026-04-23 |
| 分类 | Other \| tier: major |
| Label | ready, cpu, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/38112 |

## 一句话结论

neoverse V1 上 BF16/FP16 softmax 换用快速 exp，3-4% 加速、耗时比 1.04。

## 动机

PR 正文指出 NEON 硬件上缺少面向低精度 dtype 的更快 exp 例程。原有 `neon_expf` 来自 Arm Optimized Routines 的 expf AdvSIMD，按 FP32 精度实现（五次多项式、把 ln2 拆成 high/low 两部分）。diff 注释说明 attention softmax 的输出会被 downcast 到 FP16/BF16，此时高精度 exp 存在冗余，因此值得为「输出将 downcast 到 FP16/BF16」的场景提供更快的近似，并在 neoverse V1 上观察到 3-4% 加速。

## 优化点

在 `__aarch64__` 的 `DEFINE_FAST_EXP` 宏内新增 `neon_expf_f16` 与 `fast_exp_f16`。该快速指数用三次多项式近似 `exp(r)` 替代原来的五次多项式并重新调系数，不再把 ln2 拆成 high/low 两部分；通过 `n = round(x/ln2)`、`r = x - n*ln2`、`p = 1 + r + r^2*(c3 + c2*r)`，再把 `2^n` 以指数位拼接进结果，并把 `x < -87.683113f` 钳到 0、`x > 88.3762589f` 钳到 inf。diff 注释标注其对 FP16/BF16 在 [-87.683, 88.376] 内精度在 1 ULP 内。在 `AttentionMainLoop` 两处 softmax/softcap 的 exp 计算中，依据 `query_t` 是否为 `c10::BFloat16`/`c10::Half`（`IsReducedPrecision`）在 `__aarch64__` 下选择 `fast_exp_f16`，否则回退原 `fast_exp`；仅在 `DEFINE_FAST_EXP` 定义时生效。

## 关键改动

```text
csrc/cpu/cpu_arch_macros.h: 在 __aarch64__ 的 DEFINE_FAST_EXP 宏中新增 neon_expf_f16 / fast_exp_f16 低精度快速 exp。
csrc/cpu/cpu_attn_impl.hpp: AttentionMainLoop 两处 exp 按 IsReducedPrecision 选择 fast_exp_f16。

  ; neon_expf_f16 核心计算:
  float32x4_t n = vrndaq_f32(vmulq_f32(values, inv_ln2));
  float32x4_t r = vfmsq_n_f32(values, n, ln2);
  uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), 23);
  float32x4_t r2 = vmulq_f32(r, r);
  float32x4_t q = vfmaq_n_f32(f_c3, r, f_c2);
  float32x4_t s = vaddq_f32(vdupq_n_f32(1.0f), r);
  float32x4_t p = vfmaq_f32(s, q, r2);
  float32x4_t y = vreinterpretq_f32_u32(vaddq_u32(vreinterpretq_u32_f32(p), e));
```

## 性能声明

PR 正文 Purpose 声称「Observed a 3-4% speedup on neoverse V1」，Test Result 给出「baseline mean (ms) / faster exp = 1.04」。测试条件（Test Plan）：`LD_PRELOAD` 加载 `libtcmalloc_minimal.so.4` 与 `libgomp.so.1`、`taskset -c 20-27`，运行 `benchmark_attention_minimal.py`（复制自 `benchmark_cpu_attn.py`），配置 `batch_size=64`、`isa="neon"`、`seq_len=512`、`iters=50`、`dtype=torch.bfloat16`、`seq_lens=[(512,512)]*64`、`num_heads=(32,8)`、`head_size=128`。硬件 neoverse V1 / NEON、精度 bfloat16；PR 未注明具体模型。

## 验证与回归风险

PR 正文的 Test Plan 仅要求运行上述 attention benchmark 并对比 mean 耗时，未引入新的单元测试，也未列出回归测试套件。可能受影响的路径：新 exp 仅在 `__aarch64__` 且定义 `DEFINE_FAST_EXP`、且 query 为 BF16/FP16 时启用；FP32 路径仍走原 `fast_exp`，非 aarch64 平台不受影响。新增 exp 为带钳位的近似（FP16/BF16 1 ULP 内），对低精度 softmax 数值可能有细微影响，softcap 分支同样改走该选择逻辑。PR 未注明更广泛的回归验证。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
