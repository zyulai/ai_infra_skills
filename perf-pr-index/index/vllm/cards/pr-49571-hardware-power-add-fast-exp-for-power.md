# PR #49571: [Hardware][Power] Add FAST_EXP for Power

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Akashcodes732 |
| 合并 | 2026-07-27 |
| 分类 | Other |
| Label | performance, ready, cpu |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/49571 |

## 一句话结论

为 PowerPC (VSX) CPU 补上 `DEFINE_FAST_EXP` 向量化 exp,CPU attention 微基准快 17.4%,端到端吞吐 +4.5%。

## 动机

vLLM CPU 后端的 attention 路径通过 `DEFINE_FAST_EXP` 宏在各架构上接入快速指数实现(RISC-V 向量扩展等已有分支),但 PowerPC (`__powerpc__`) 没有对应定义,CPU attention 中的 `expf`(softmax 指数)只能走慢路径。本 PR 用 VSX intrinsics 补齐 Power 的向量化 exp,使 CPU attention 基准在对齐其他架构的快路径后提速。

## 优化点

在 `csrc/cpu/cpu_arch_macros.h` 新增 `__powerpc__` 分支的 `DEFINE_FAST_EXP`:将 `fast_exp` lambda 内联到 `FP32Vec16::exp()`,后者拆成上下两个 `FP32Vec8` 分别计算再拼回;`FP32Vec8::exp()` 在 VSX 上用 5 项 minimax 多项式向量化实现(依据 diff 中注释)。同时把 `FP32Vec4/8/16` 的拷贝构造函数从 `explicit` 放开,使 `exp()` 内的中间构造得以编译。另修了 CPU attention 基准脚本的一处 bug:`sliding_window` 无窗口时应传 `-1` 而非把 `window_size=None` 直接透传。

## 关键改动

```text
csrc/cpu/cpu_arch_macros.h: 新增 __powerpc__ 的 DEFINE_FAST_EXP 宏
csrc/cpu/cpu_types_vsx.hpp: 新增 FP32Vec16::exp();FP32Vec4/8/16 拷贝构造去 explicit
benchmarks/kernels/cpu/benchmark_cpu_attn.py: 修复 sliding_window=None 时应传 -1
```

```cpp
// Power VSX
#ifdef __powerpc__
  // FP32Vec16::exp() in cpu_types_vsx.hpp delegates to FP32Vec8::exp(), which
  // implements a vectorised 5-term minimax polynomial using VSX intrinsics.
  #define DEFINE_FAST_EXP                                                     \
    auto fast_exp = [&](const vec_op::FP32Vec16& vec)                         \
                        __attribute__((always_inline)) { return vec.exp(); }; \
    auto fast_exp_f16 = fast_exp;
#endif  // __powerpc__
```

## 性能声明

CPU attention 微基准(`python benchmarks/kernels/cpu/benchmark_cpu_attn.py`):Mean Time 249.15 ms → 205.75 ms,1.21x(17.4% faster);Min/Max 分别 248.61→205.18 / 249.72→206.47 ms。硬件为 PowerPC CPU,PR 未注明具体 Power 机型与核数。

端到端吞吐(`vllm bench throughput --model ibm-granite/granite-3.1-8b-instruct --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100 --dtype bfloat16`):Total Tokens/s 86.24 → 90.13(+4.5%),Output Tokens/s 42.24 → 44.14(+4.5%),Requests/s 0.17 → 0.18(+5.8%)。

## 验证与回归风险

验证仅由上述两个基准构成,PR 未附单元测试或数值精度对比(minimax 多项式近似相对 `expf` 的误差 PR 未注明)。改动全部圈定在 `__powerpc__` 编译分支内,其他架构(x86/RISC-V 等)不受影响;VSX 类型文件的 `explicit` 拷贝构造放开会作用于整个 VSX 编译单元,属于潜在的编译期行为变化。风险面较小:CPU 路径专用,CUDA/ROCm 主路径无关。

## 关联

同区域(CPU perf):索引内 PR #25948(CPU 未量化 linear 加速)、PR #43534(GDN 系列启用 fused kernels,CPU 相关)。机制上同类:各架构 `DEFINE_FAST_EXP` 快速指数分支(RISC-V V 扩展已有实现,本 PR 补 Power VSX)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
