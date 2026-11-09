# PR #4944: [Fix] revert clean m.def for cudagraph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4944](https://github.com/sgl-project/sglang/pull/4944) |
| 作者 | FlamingoPg |
| 合并日期 | 2025-03-31 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
恢复 #4890 清理掉的带 schema `m.def`:无 schema 定义导致 cudagraph/torch.compile 失败,并为各 kernel 补回 `m.impl` 设备绑定与文档。

## 问题现象
#4890 清理 `m.def` 之后,cudagraph/torch.compile 失败(正文:"it will cause cudagraph/torch.compile fail")。正文未给出具体报错堆栈或复现命令。

## 根因
正文披露:`m.def` 带 schema 能帮助 torch inductor 找到 custom ops("Cauz m.def with schema will help torch inductor to find out custom ops")。#4890 把 `torch_extension.cc` 中的注册简化为无 schema 的函数指针 `m.def("name", fn)`,使 torch.compile/inductor 无法按 schema 识别与自动捕获这些自定义算子,cudagraph 捕获随之失败。

## 修复方案
`sgl-kernel/csrc/torch_extension.cc` 为 allreduce/attention/elementwise/gemm/moe/speculative/sampling/flash-attention 等各 kernel 恢复「带 schema 的 `m.def` + `m.impl(name, torch::kCUDA, fn)`」形式,例如:

```cpp
m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, int cuda_stream) -> ()");
m.impl("rmsnorm", torch::kCUDA, &rmsnorm);
```

同时更新 `sgl-kernel/README.md`:说明 torch.compile 需要 `m.def` 带 schema、给出 schema 书写参考,以及 flash-attention `fwd` 用 `m.impl` + `make_pytorch_shim` 绑定、TORCH_LIBRARY 需要 `double`/`int64_t` 的原因。

## 影响与回归风险
- 影响窗口:#4890 合并至本修复之间。
- 回归面:sgl-kernel 全部算子的 Python 绑定声明——签名、参数名与默认值改由 schema 决定;按位置传参或依赖 #4890 新形式的调用方需核对;文档同步更新,属于绑定层/API 面改动而非 kernel 实现改动。

## 关联
- [#4890](https://github.com/sgl-project/sglang/pull/4890)(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
