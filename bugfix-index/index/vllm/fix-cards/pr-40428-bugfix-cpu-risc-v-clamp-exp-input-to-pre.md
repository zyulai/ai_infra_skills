# PR #40428: [Bugfix][CPU][RISC-V] Clamp exp() input to prevent NaN

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40428](https://github.com/vllm-project/vllm/pull/40428) |
| 作者 | lyd1992 |
| 合并日期 | 2026-04-22 |
| labels | bug, cpu, ready, verified |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
RISC-V 向量 `FP32Vec8/FP32Vec16::exp()` 对大输入在多项式下溢后出现 -inf*0=NaN,softmax 输出 NaN;求值前把输入钳位到 [-87.33, 88.72] 修复。

## 问题现象
在 RISC-V(RVV)CPU 路径上,`exp()` 对大数值输入产生 NaN,进而在 attention 中表现为 softmax NaN。正文说明该问题是作者在调试 RVV attention 路径的 softmax NaN 时发现的。具体影响面由根因决定:任何把较大值(如 -inf 或超出 exp 定义域的值)送入该 exp 实现的计算,包括 softmax 中对极低分数求 exp 的场景。

## 根因
正文披露:`csrc/cpu/cpu_types_riscv_impl.hpp` 中 `FP32Vec16::exp()`(以及同构的 `FP32Vec8::exp()`,见 diff)在输入较大时产生 NaN。机制是:输入经过 `* inv_ln2` 与多项式近似后,多项式项下溢为 0,而缩放因子为 -inf 时按 IEEE 754 规则 `-inf * 0.0 = NaN`,最终结果为 NaN。`exp(-inf)` 本应返回 0。代码注释指出 x86 AVX-512 与 ARM NEON 实现均已使用钳位策略,仅 RISC-V 实现缺失。

## 修复方案
改 `csrc/cpu/cpu_types_riscv_impl.hpp` 的 `FP32Vec8::exp()` 与 `FP32Vec16::exp()`:
1. 在多项式求值之前,用 RVV 向量指令把输入钳位到 `[exp_lo, exp_hi]`:
```python
constexpr float exp_lo = -87.3365447505f;  # ln(FLT_MIN)
constexpr float exp_hi = 88.7228391117f;   # ln(FLT_MAX)
x = vfmin(vfmax(reg, exp_lo), exp_hi)
```
(LMUL_256 对应 FP32Vec8,LMUL_512 对应 FP32Vec16。)
2. 后续 `x_scaled = x * inv_ln2` 改用钳位后的 `x` 而非原始 `reg`;其余 n 整数转换、多项式求值逻辑不变。
3. 同文件附带改动:`VLLM_DISPATCH_CASE_FLOATING_TYPES` 宏不再按 `RISCV_BF16_SUPPORT` 条件区分,统一包含 BFloat16 分支——注释说明 BFloat16 在 RISC-V 上总是支持(有原生支持时原生走,否则走 FP32 模拟回退路径),即未定义该宏的平台此前根本无法分发 BFloat16 核。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:RISC-V CPU 后端的 exp 计算(softmax/attention 等依赖 exp 的路径)与浮点 dtype 分发宏。钳位只影响超出 [-87.34, 88.73] 的输入,把这些输入从 NaN(或溢出 inf)变为正确的 0/饱和值,与 x86/ARM 行为对齐;正常范围内的输入结果不变。分发宏改动使原本编译不到的 BFloat16 分支在所有 RISC-V 构建上可用,若某平台 BF16 回退路径本身有问题会新暴露(正文称 BFloat16 总是被支持,labels 含 verified)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
