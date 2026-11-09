# PR #34484: [ROCm] Fix QuickReduce fp16 saturation corrupting bf16 all-reduces (106M non-finite -> 0, +0.3%)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34484](https://github.com/sgl-project/sglang/pull/34484) |
| 作者 | alexnails |
| 合并日期 | 2026-08-30 |
| labels | sgl-kernel, run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

QuickReduce 的 bf16→fp16 转换在超 65504 时产生 inf 污染 GPT-OSS all-reduce;加 FP16_OVFL 钳制与 2 的幂缩放范围防护修复。

## 问题现象

GPT-OSS 模型的激活幅值会超过 FP16 最大有限值 65,504。QuickReduce 为性能先把 BF16 输入转成 FP16 再做 all-reduce,转换产生 `inf`;量化编解码器随后把这个非有限值通过 32 元素块传播开,导致 GPT-OSS 的 all-reduce 输出全部为非有限值(标题记录:106M non-finite,修复后归零)。修复前该问题对应 issue #34473。

## 根因

按 PR 正文:

- bf16→fp16 快速转换路径缺少范围防护,幅值超过 FP16 上限 65,504 的输入直接变成 `inf`。
- HIP 内建转换下,LLVM 会把 `(bf16_as_f32 * scale) -> fp16` 重关联成 `fp16(bf16_as_f32) * scale`,即在范围防护(缩放)生效之前就把超限值裁成 fp16,依然得到非有限值(正文对编译器行为的解释,diff 注释同步说明)。
- MODE.FP16_OVFL 此前仅在 gfx942 上启用,gfx950 上溢出仍产生 inf(由 diff 推断:`set_fp16_ovfl` 原 `#if defined(__gfx942__)` 条件不含 gfx950)。

## 修复方案

改 `python/sglang/kernels/aot/csrc/allreduce/quick_all_reduce.cuh` 与 `quick_all_reduce_base.h`,新增四 GPU 回归测试:

- 在 gfx942 与 gfx950 上都启用 `MODE.FP16_OVFL`,FP16 溢出钳制到 ±MAX_FP16 而非 inf;为 `s_setreg` 内联汇编补 `"memory"` clobber,防止编译器把转换提升到 `s_setreg` 之前。
- 为 `CodecFP` 引入 2 的幂范围防护 `S = 16`(`kQRFp16CastScaleLog2Fp = 4`):加载侧在 FP32 里除以 S 再收窄成 FP16,输出侧乘回 S,位移精确故 `sum(x_i / S) * S == sum(x_i)`。
- 用显式 `v_cvt_pk_f16_f32` 指令实现缩放转换,阻止 LLVM 把 FP16 转换移到缩放之前:

  ```diff
  +__quickreduce_device_inline__ half2 scaled_bfloat162_to_half2(nv_bfloat162 value, float scale) {
  +  float2 scaled = __bfloat1622float2(value);
  +  scaled.x *= scale;
  +  scaled.y *= scale;
  +  int packed;
  +  asm volatile("v_cvt_pk_f16_f32 %0, %1, %2" : "=v"(packed) : "v"(scaled.x), "v"(scaled.y));
  +  return *reinterpret_cast<half2*>(&packed);
  +}
  ```

- `CodecQ4/Q6/Q8` 保持 `S = 1` 走既有快速路径:它们已有块缩放做归一化,提高 S 会把倒数缩放饱和悬崖推入真实数据(diff 注释记录:S 从 1 提到 2 时 GLM-5.2 上困惑度损失 62%)。
- 新增 `test/registered/kernels/test_quick_allreduce_bf16_range.py`:四 GPU 覆盖低值、普通值、和超 FP16 范围、单输入超 FP16 范围、负向溢出、混合块等用例。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;问题影响使用 ROCm QuickReduce bf16→fp16 快速路径(如 GPT-OSS)的多卡推理。
- 回归面:ROCm QuickReduce 全部量化模式(FP/INT8/INT6/INT4)的 bf16 all-reduce 数值路径。量化编解码在 FP16 累加器超 65,504 时仍可能钳制有限值,属既有有损行为,本次有意不改;原生 BF16 方案数值精确但慢 22–46%,未被采用。实测 MI350X 上 INT8/INT6/INT4 在 16 MiB 差 0.4% 内、90 MiB 差 0.1% 内,FP 路径本次运行反而快 9.6%(16 MiB)/1.1%(90 MiB)。

## 关联

- #34473(被修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
