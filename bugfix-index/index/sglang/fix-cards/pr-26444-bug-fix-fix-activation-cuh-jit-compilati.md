# PR #26444: [Bug Fix] Fix activation.cuh JIT compilation failure on CUDA 13 due to template type/value mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26444](https://github.com/sgl-project/sglang/pull/26444) |
| 作者 | 256256mjw |
| 合并日期 | 2026-06-05 |
| labels | jit-kernel, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
CUDA 13 的 nvcc 更严格,`activation.cuh` 中 `select_kernel` 的 decltype 尾置返回类型把枚举值误当类型解析,JIT 编译失败使服务崩溃;改用显式 `kernel_fn_t` 别名修复。

## 问题现象
在 CUDA 13.0 上,`activation.cuh` 的 JIT 编译在 CUDA graph capture 期间失败,导致服务进程崩溃(修复 issue #25487)。编译报错原文:

```
activation.cuh:95:52: error: type/value mismatch at argument 1 in template parameter list
  note: expected a type, got 'ActivationKind::kSiLU'
```

## 根因
正文披露:根因是 `select_kernel` 尾置返回类型中的 C++ 模板 type/value 不匹配。CUDA 13 更严格的 `nvcc` 无法在依赖上下文(dependent context,含外层模板参数 `kFilterExpert` 的 trailing return type)中正确解析 `decltype(ActivationKernel::template activation_kernel<ActivationKind::kSiLU, kFilterExpert>)`,把非类型的枚举值 `ActivationKind::kSiLU` 误解释为类型实参;旧版 CUDA(11/12)会静默接受这种写法。

## 修复方案
全部为编译期修改,改 `python/sglang/jit_kernel/csrc/elementwise/activation.cuh`(正文列出 5 点,diff 逐一对应):

1. 用 `decltype(&act_and_mul_kernel<T, ActivationKind::kSiLU, kUsePDL, false>)` 在全局函数模板上引入显式 `kernel_fn_t` 类型别名,绕开有问题的变量模板解析路径;
2. `activation_kernel` 变量模板的 `auto` 改为 `kernel_fn_t`;
3. `select_kernel` 去掉 decltype 尾置返回类型,直接返回 `kernel_fn_t`;
4. 移除类内不必要的 `ActivationKernel::template` 限定;
5. `hidden_size` 加 `static_cast<uint32_t>()` 消除有符号→无符号收窄警告。

关键 diff 摘录:

```diff
+  using kernel_fn_t = decltype(&act_and_mul_kernel<T, ActivationKind::kSiLU, kUsePDL, false>);
+
   template <ActivationKind kAct, bool kFilterExpert>
-  static constexpr auto activation_kernel = act_and_mul_kernel<T, kAct, kUsePDL, kFilterExpert>;
+  static constexpr kernel_fn_t activation_kernel = act_and_mul_kernel<T, kAct, kUsePDL, kFilterExpert>;
```

```diff
   template <bool kFilterExpert>
-  static auto select_kernel(const std::string& type)
-      -> decltype(ActivationKernel::template activation_kernel<ActivationKind::kSiLU, kFilterExpert>) {
+  static kernel_fn_t select_kernel(const std::string& type) {
```

兼容 CUDA 11、12、13 与 ROCm/HIP;生成 PTX/SASS 完全相同(函数指针类型与 kernel 代码未变),无运行时行为或性能变化。

## 影响与回归风险
- 影响窗口:正文未说明,不写(问题在 CUDA 13.0 上触发)。
- 回归面:仅 `activation.cuh` 的编译期代码:silu/gelu/gelu_tanh 三种激活的 kernel 选择路径现在共享同一 `kernel_fn_t` 签名,若未来某激活 kernel 签名不同会在此处编译报错(显式失败,优于原先的隐式解析);`hidden_size` 的显式转换不影响数值。纯编译修复,模型输出与性能无变化。

## 关联
- https://github.com/sgl-project/sglang/issues/25487(正文开头给出的 issue 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
