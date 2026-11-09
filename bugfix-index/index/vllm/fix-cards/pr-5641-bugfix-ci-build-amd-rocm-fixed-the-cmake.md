# PR #5641: [Bugfix][CI/Build][AMD][ROCm]Fixed the cmake build bug which generate garbage on certain devices

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5641](https://github.com/vllm-project/vllm/pull/5641) |
| 作者 | hongxiayang |
| 合并日期 | 2024-06-19 |
| labels | (none) |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm cmake 用错 HIP 目标架构(误用构建机的 gfx906)导致 MI300x 输出垃圾,修复为遍历官方支持的 gfx908/90a/942/1100 多架构列表。

## 问题现象
客户报告 vllm v0.4.0 之后在 MI300x + ROCm 6.1 docker(`rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2`)上推理输出垃圾(garbage)。构建日志显示 `-- HIP target arches: gfx906`、`-- Punica target arches: gfx906`,即内核被编给错误架构;修复后正确显示 `-- HIP target arches: gfx908;gfx90a;gfx942;gfx1100`,MI300x 不再产生垃圾。正文列出该 bug 的三重危害:(1) 正确性——vllm CudaExtension 内核、Punica 内核、MoE 内核都按错误架构编译,MI300x(gfx942)上行为异常;(2) 可扩展性——只为一个 GPU 目标编译,无法一次构建多架构镜像跨平台复用;(3) 基准测试可能产生错误数字。

## 根因
正文披露:作者在 MI300x 上用 main 分支调试数天后确认,当前 cmake 未使用正确的 HIP target(GPU 架构)——`override_gpu_arches` 宏遍历的是 `CMAKE_HIP_ARCHITECTURES`,在构建机上解析为 gfx906,而非目标设备架构,导致所有 ROCm 内核按错误 arch 编译,在 gfx942 上运行即产生垃圾输出。

## 修复方案
- `cmake/utils.cmake` 的 `override_gpu_arches` 宏:不再遍历 `CMAKE_HIP_ARCHITECTURES`,改为遍历新增的 `VLLM_ROCM_SUPPORTED_ARCHS = "gfx908;gfx90a;gfx942;gfx1100"`,与 `_GPU_SUPPORTED_ARCHES_LIST` 求交集:

```cmake
set(VLLM_ROCM_SUPPORTED_ARCHS "gfx908;gfx90a;gfx942;gfx1100")
set(${GPU_ARCHES})
foreach (_ARCH ${VLLM_ROCM_SUPPORTED_ARCHES})
  if (_ARCH IN_LIST _GPU_SUPPORTED_ARCHES_LIST)
    list(APPEND ${GPU_ARCHES} ${_ARCH})
  endif()
```

- `Dockerfile.rocm` 重构(按 @WoosukKwon 反馈):BASE_IMAGE 对比改为参数化变量(`ROCm_5_7_BASE`/`ROCm_6_0_BASE`);rocm_bf16.patch 仅在 ROCm 6.0 基础镜像上打;拷贝 `.so` 时按 `VLLM_PYTHON_VERSION` 动态确定路径并一次拷贝 `vllm/*.so`。

## 影响与回归风险
- 回归面:所有 ROCm 构建路径——vllm 主内核、Punica、MoE 的编译架构集合从单一构建机架构变为固定四架构列表与支持列表的交集;若某设备 arch 不在该列表(如 gfx906 卡)将不再被编译,属有意收窄;Dockerfile.rocm 的 bf16 patch 行为按基础镜像版本区分,ROCm 6.0 前版本不再打该 patch。

## 关联
正文未列出(正文仅有「FIX #xxxx」占位符,未给出真实 issue 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
