# PR #41387: [Fix] Add missing stubs from cpu fp8 attention changes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41387](https://github.com/vllm-project/vllm/pull/41387) |
| 作者 | tianmu-li |
| 合并日期 | 2026-05-06 |
| labels | ready, cpu, verified |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Quantization |
| bug 类型 | hang |

## 一句话结论
PR #39445 的 CPU fp8 attention 改动漏掉了非 x86 平台向量的 FP8 stub 构造函数,clang 下编译报错;为 ARM/scalar/VSX/VXE 四个路径补上空实现 stub。

## 问题现象
使用 clang 编译 vLLM CPU 后端时出现编译错误(正文:"cause compilation errors when using clang",并附作者 fork 仓库 CI 的报错日志链接)。原因是 PR #39445 引入的 CPU fp8 attention 改动缺少部分 stub(正文:"is missing some stubs for fp8 attention")。影响面为非 x86 的 CPU 平台构建(ARM、scalar、PowerPC/VSX、s390x/VXE),x86 平台不受影响。

## 根因
正文披露:PR #39445 为 CPU fp8 attention 引入了新模板代码(如 `load_b_pair_vec`),但没有为所有平台向量类型头文件补齐所需的构造函数 stub。diff 中新增的注释直接说明了机制:fp8 KV cache 仅 x86 支持,这些构造函数在 ARM/scalar/PowerPC/s390x 上是死代码("FP8 stub: dead code on ... fp8 KV cache is x86-only"),但模板实例化要求它们在所有平台上都能编译通过,缺失即编译失败。

## 修复方案
在 4 个 CPU 向量类型头文件中补上 stub 构造函数(均为空实现或零初始化,不做真实 fp8 转换):
- `csrc/cpu/cpu_types_arm.hpp`:`FP32Vec16(const BF16Vec32&, int)` 空构造;
- `csrc/cpu/cpu_types_scalar.hpp`:新增 `fp8_e4m3_tag`/`fp8_e5m2_tag` 两个空 tag 结构,`BF16Vec32(const uint8_t*, fp8_e4m3_tag/e5m2_tag)` 零初始化构造,以及 `FP32Vec16(const BF16Vec32&, int)`;
- `csrc/cpu/cpu_types_vsx.hpp`(PowerPC):同款 `BF16Vec32` fp8 tag 构造与 `FP32Vec16(const BF16Vec32&, int)`;
- `csrc/cpu/cpu_types_vxe.hpp`(s390x):`FP32Vec16(const BF16Vec32&, int)`。

```diff
+  // FP8 stub: dead code on ARM (fp8 KV cache is x86-only), needed for
+  // load_b_pair_vec template to compile on all platforms.
+  explicit FP32Vec16(const BF16Vec32&, int) : Base() {}
```

## 影响与回归风险
- 影响窗口:正文指出缺口由 PR #39445 引入;具体版本号未提及。
- 回归面:仅非 x86 CPU 平台的向量头文件新增空 stub 构造函数,不影响 x86 运行时行为;stub 是显式写出的死代码,风险在于若未来这些平台真的支持 fp8 KV cache,空实现可能被误调用(当前均为不可达路径)。labels 含 verified,作者 CI(clang)已验证可编译。

## 关联
- 引入问题的 PR: https://github.com/vllm-project/vllm/pull/39445
- 编译报错日志: https://github.com/tianmu-li/vllm/actions/runs/25149522905/job/73716695058#logs

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
