# PR #34484: [ROCm] Fix QuickReduce fp16 saturation corrupting bf16 all-reduces (106M non-finite -> 0, +0.3%)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @alexnails |
| 合并 | 2026-08-30 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | sgl-kernel, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,3 个文件,含 1 个新测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/34484 |

## 一句话结论

修复 QuickReduce bf16 all-reduce 在 GPT-OSS 大数值下 fp16 饱和产生 inf 并污染 32 元素块的 bug:106M 非有限输出降为 0,且 FP 路径反而更快。

## 动机

QuickReduce(ROCm 快速 all-reduce)为性能把 BF16 输入先转 FP16 再做归约。GPT-OSS 模型的激活幅度超过 FP16 最大有限值 65,504,转换产生 `inf`;量化 codec 随后把该值经块内传播,导致整块输出非有限(issue #34473,标题实测 106M non-finite -> 0)。数值精确的 native-BF16 替代方案要慢 22-46%,不可接受,需要保住 FP16 快路径的同时修掉溢出。

## 优化点

三层修复:(1) 在 gfx942 与 gfx950 上都启用 `MODE.FP16_OVFL`(`s_setreg_imm32_b32`),使 FP16 溢出钳位到 +/-65504 而非 inf,并给 setreg 加 "memory" clobber 防止编译器把转换提升到 setreg 之前;(2) 为无块 scale 的 `CodecFP` 引入 2 的幂范围防护 S=16(`kQRFp16CastScaleLog2Fp=4`):加载侧在 FP32 除以 16 再收窄到 FP16,输出侧乘回——2 的幂缩放精确,故 `sum(x_i/S)*S == sum(x_i)`;(3) 用内联 `v_cvt_pk_f16_f32` 显式发射缩放后的 CodecFP 转换,阻止 LLVM 把 bf16->fp16 转换重排到 scale 之前(那会先在 65504 处 clip)。量化 codec `CodecQ4/Q6/Q8` 保持 S=1:它们已有 32 值块 scale 归一化,提高 cast scale 会把 reciprocal-scale 饱和悬崖推进真实数据(注释:S 从 1 提到 2 在 GLM-5.2 上实测 62% 困惑率)。

## 关键改动

```text
python/sglang/kernels/aot/csrc/allreduce/quick_all_reduce_base.h: 常量定义 S=16(FP)/S=1(量化)与 set_fp16_ovfl 扩展到 gfx950 + memory clobber。
python/sglang/kernels/aot/csrc/allreduce/quick_all_reduce.cuh: 各 codec 的 kCastScaleLog2、scaled_bfloat162_to_half2 内联转换、twoshot 加载/存储侧缩放。
test/registered/kernels/test_quick_allreduce_bf16_range.py: 新增,4 GPU 回归测试(FP/INT8/INT6/INT4 x 6 种数值场景)。
```

范围防护核心:

```cpp
__quickreduce_device_inline__ half2 scaled_bfloat162_to_half2(nv_bfloat162 value, float scale) {
  float2 scaled = __bfloat1622float2(value);
  scaled.x *= scale; scaled.y *= scale;      // FP32 侧先缩放
  asm volatile("v_cvt_pk_f16_f32 %0, %1, %2" ...)  // 显式收窄,防重排
}
// 存储侧: f *= kCastScale;  fp32 中间量不会溢出
```

## 性能声明

MI350X 上 PR-base 与最终补丁的延迟对比:INT8/INT6/INT4 在 16 MiB 内 0.4%、90 MiB 内 0.1%(无可测回退);FP 路径本次运行在 16 MiB 快 9.6%、90 MiB 快 1.1%(来自 v_cvt_pk_f16_f32 显式发射与路径简化,标题另提及整体 +0.3%)。正确性收益:捕获的 GPT-OSS 归约输出从非有限变为 0 个非有限。测试条件:TP4,ROCm 7.2。原文未提供端到端模型吞吐数字。

## 验证与回归风险

验证:TP4 range regression 在 MI350X 与 MI355X(ROCm 7.2)通过;`CodecFP` 全部用例 bit-exact(含输入与和超出 FP16 范围);量化 codec 低值/常规值 bit-exact、溢出场景输出有限;新增四卡回归测试覆盖 low/ordinary/sum 超限/单输入超限/负溢出/混合块 6 种场景 x 4 种 quant_mode,注册到 AMD CI(stage-c-4-gpu-amd 与 mi35x)。PR 明确声明遗留行为:量化归约在 FP16 累加器超 65504 时仍可能钳位有限值,属既有有损行为,本 PR 不改。pack 中 CI:Base 通过,Extra 与 AMD ROCm 7.2 标红。风险:S=16 只护 CodecFP,量化路径的饱和悬崖属设计取舍;`v_cvt_pk_f16_f32` 为 GCN 专有 ISA 指令,仅 ROCm 路径生效。

## 关联

同区域(ROCm 通信/AllReduce):PR #22734(MSCCL integration)、PR #23927(AMD FP8 MLA);issue #34473 为直接前因。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
