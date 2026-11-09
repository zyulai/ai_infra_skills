# PR #61844: [revert] [Build] Fix QuTLASS compilation with PyTorch 2.13

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61844](https://github.com/vllm-project/vllm/pull/61844) |
| 作者 | cleonard530 |
| 合并日期 | 2026-10-27 |
| labels | ready, ci/build |
| 热度 | 👍 0 · 💬 22 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | regression |

## 一句话结论
回退 #58173 的 QuTLASS 编译改动,并在 CMake 里校验 QuTLASS 源已含 stable ABI(registration.h),防止旧源混入新编译旗标。

## 问题现象
QuTLASS 在 PyTorch 2.13 下编译出问题(本 PR 标题即「Fix QuTLASS compilation with PyTorch 2.13」的回退)。正文用 torch-abi-audit 展示 main 分支的审计结果异常,而本分支构建成功并输出 `[STABLE  ] [abi3-ok] _qutlass_C.abi3.so  (stable_shim=60, unstable=0)  min-torch=2.11.0`。正文未给出面向用户的报错日志原文;这是构建/CI 层面的回归,不影响推理数值路径。

## 根因
正文披露(作者自述其判断):发生的事情是,一个迁移到 stable ABI 之前的本地旧版 QuTLASS,被拿去配合 #58173 新加的、本意是保障 QuTLASS 稳定性的编译旗标一起构建,二者不匹配导致编译问题。作为补强,本 PR 增加了版本检查:`qutlass/csrc/include/registration.h` 这个文件只在该迁移之后才存在,以此判定源码是否为迁移后的版本。

## 修复方案
改 `cmake/external_projects/qutlass.cmake`,两件事:
1. 回退 #58173 的改动:编译定义中移除 `QUTLASS_DISABLE_PYBIND=1`,新增 `TORCH_TARGET_VERSION=0x020D000000000000ULL`;
2. 新增源码版本守卫,源目录缺该头文件即 FATAL_ERROR:

```cmake
+if(NOT EXISTS "${qutlass_SOURCE_DIR}/qutlass/csrc/include/registration.h")
+  message(FATAL_ERROR
+    "[QUTLASS] Stable ABI support was not found in the Qutlass source. "
+    "Please update your Qutlass package.")
+endif()
```

验证:本分支构建成功,torch-abi-audit 显示 `_qutlass_C.abi3.so` 为 abi3-ok、unstable=0、min-torch=2.11.0;正文注明 stable_shim 数量下降源于 torch/stable 在 2.13 中的内部实现差异(新增 `torch_exception_get_what` 等接口),并非本改动引入。

## 影响与回归风险
- 影响窗口:自 #58173 合入起,至本回退前;使用 QuTLASS 且构建环境存在旧版源码(迁移前)时触发。
- 回归面:仅 CMake 构建脚本,不触碰运行时代码。两点行为变化:其一,编译定义回到 #58173 之前的组合并显式指定 TORCH_TARGET_VERSION;其二,构建期新增硬失败检查——环境里装着旧版 QuTLASS 的用户将从"编译出诡异结果"变为"明确报错要求升级包",属预期的前置失败。

## 关联
- [#58173](https://github.com/vllm-project/vllm/pull/58173)(被回退的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
