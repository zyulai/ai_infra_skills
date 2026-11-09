# PR #33992: [Bugfix] Fix CUDA compatibility path setting for both datacenter and consumer NVIDIA GPUs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33992](https://github.com/vllm-project/vllm/pull/33992) |
| 作者 | ehfd |
| 合并日期 | 2026-02-26 |
| labels | bug, ci/build, documentation, nvidia, ready |
| 热度 | 👍 1 · 💬 36 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
官方镜像无条件加载 CUDA compat 库导致 GeForce 等消费级 GPU 报错,改为按 VLLM_ENABLE_CUDA_COMPATIBILITY 环境变量选择性启用。

## 问题现象
vllm/vllm-openai 官方容器中,所有 GeForce(消费级)GPU 用户都遇到报错;典型表现为 NVIDIA 驱动过旧时报 `RuntimeError: The NVIDIA driver on your system is too old (found version 12090)`,而 compat 库被错误加载时消费级 GPU 上出现 `RuntimeError: Unexpected error from cudaGetDeviceCount() ... Error 803: system has unsupported display driver / cuda driver combination`。正文指出两条原则:消费级 NVIDIA GPU 绝不能把 CUDA compatibility 库放进 `LD_LIBRARY_PATH`;专业卡/数据中心卡只在特定情况下支持,应选择性启用。除改 vLLM 可执行文件/手动改 `LD_LIBRARY_PATH` 外别无他法,非常不直观。

## 根因
正文披露:问题由 #30784 与 #33116 引入——Dockerfile 里无条件执行 `RUN echo "/usr/local/cuda-<ver>/compat/" > /etc/ld.so.conf.d/cuda-compat.conf && ldconfig`,强制让所有 GPU(含不支持 CUDA forward compatibility 的消费级卡)都优先加载 compat 库,从而在消费卡上触发 Error 803,且没有开关可关。

## 修复方案
- `docker/Dockerfile`:两处删除上述无条件 ldconfig 写入,改为默认 `ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0`。
- `vllm/env_override.py`:新增 `_get_torch_cuda_version()`(不 import torch、避免提前触发 CUDA 初始化)与 `_maybe_set_cuda_compatibility_path()`,在 `import torch` 之前按环境变量解析:仅当 `VLLM_ENABLE_CUDA_COMPATIBILITY` 为 `1/true` 时,按「自定义路径 > `$CONDA_PREFIX/cuda-compat` > `/usr/local/cuda-{ver}/compat`」探测 compat 目录,去重后前插到 `LD_LIBRARY_PATH` 首位,已在首位则不动。
- `vllm/envs.py`:注册 `VLLM_ENABLE_CUDA_COMPATIBILITY`、`VLLM_CUDA_COMPATIBILITY_PATH` 并纳入 compile_factors。
- 新增 `tests/cuda/test_cuda_compatibility_path.py`(21 个单测,`21 passed in 1.25s`),并更新安装与 troubleshooting 文档。

## 影响与回归风险
- 回归面:Docker 镜像构建与启动环境(不再默认写 /etc/ld.so.conf.d/cuda-compat.conf)、vLLM 进程最早期的 env_override 导入路径(现在会操作 LD_LIBRARY_PATH)。数据中心/专业卡用户需显式传 `-e VLLM_ENABLE_CUDA_COMPATIBILITY=1`(必要时配 `VLLM_CUDA_COMPATIBILITY_PATH`),消费卡保持 0;正文验证了 A100 与 Quadro RTX 6000 在驱动 575/580 + CUDA 12.9/13.0 组合下的开/关行为。依赖旧镜像默认加载 compat 行为的用户属于行为变更。

## 关联
Fixes #32373、#33369、Closes #34226;问题由 #30784、#33116 引入

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
