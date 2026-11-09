# PR #12375: [Bugfix][Kernel] Fix CUDA 11.8 being broken by FA3 build

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#12375](https://github.com/vllm-project/vllm/pull/12375) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-01-24 |
| labels | documentation, ready, ci/build |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
FA3 依赖 CUDA 12+ 特性导致 CUDA 11.8 构建失败;11.8 构建下不再编译 FA3 扩展并完善版本检查报错。

## 问题现象
CUDA 11.8 构建被 FA3 破坏。正文原文:「Can build FA3 with CUDA version < 12.0 due to use of `__viaddmin_s32` and compute capability 9.0a, disable FA3 for 11.8 builds」——结合修复内容,指 FA3 在 CUDA < 12.0 下无法构建(diff 注释也写明「FA3 requires CUDA 12.0 or later」)。正文未附具体编译报错日志。

## 根因
正文披露:FA3 使用了 `__viaddmin_s32` 指令并要求 compute capability 9.0a,因此 CUDA 12.0 以下的工具链无法构建 FA3,导致 11.8 构建整体失败。

## 修复方案
- `setup.py`:仅当 `envs.VLLM_USE_PRECOMPILED` 或 nvcc 版本 >= 12.0 时才把 `vllm.vllm_flash_attn._vllm_fa3_C` 加入 ext_modules,11.8 源码构建不再编译 FA3。
- `CMakeLists.txt`:vllm-flash-attn 的 `GIT_TAG` 从 `90eacc1a…` 更新为 `0aff05f5…`。
- `vllm/attention/backends/flash_attn.py` 与 `vllm/v1/attention/backends/flash_attn.py`:`__init__` 中在不支持时先用 logger.error 输出原因(`fa_version_unsupported_reason`),再触发原有 `assert is_fa_version_supported(...)`,改善可诊断性。
- `tests/kernels/test_flash_attn.py`、`tests/kernels/test_cascade_flash_attn.py`:把硬编码的「8.6/8.9 共享内存不足则 skip」改为统一的 `is_fa_version_supported` + 原因字符串 skip。

关键 diff 摘录(setup.py):
```diff
-    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
+    if envs.VLLM_USE_PRECOMPILED or get_nvcc_cuda_version() >= Version("12.0"):
+        # FA3 requires CUDA 12.0 or later
+        ext_modules.append(
+            CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:CUDA < 12.0 的源码构建不再产出 FA3 扩展(FA2 不受影响);FA 后端初始化新增错误日志;预编译版本(`VLLM_USE_PRECOMPILED`)路径保持编译 FA3;vllm-flash-attn 依赖 tag 变更影响 FA2/FA3 两个内核的源码。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
