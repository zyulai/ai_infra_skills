# PR #34644: [release 2.11] Update to torch 2.11

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @atalman |
| 合并 | 2026-04-08 |
| 分类 | Other \| tier: major |
| Label | documentation, performance, rocm, ready, ci/build, v1, llama, cpu, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未见「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/34644 |

## 一句话结论

release 2.11 依赖升级：torch 2.10→2.11、CUDA 12.9.1→13.0.0 及配套 CI / requirements 适配。

## 动机

PR 正文仅一句 "Update to Torch 2.11"，未注明更细的动机、瓶颈或性能问题。属于 release 分支的依赖与构建配置升级，不涉及新算法或新 kernel。

## 优化点

从 diff 可见的改动是版本与平台适配：`CMakeLists.txt` 中 `TORCH_SUPPORTED_VERSION_CUDA` / `TORCH_SUPPORTED_VERSION_ROCM` 由 2.10.0 升到 2.11.0；`docker/Dockerfile` 的 `CUDA_VERSION` 由 12.9.1 升到 13.0.0、`BUILD_BASE_IMAGE` 由 ubuntu20.04 改为 ubuntu22.04，并调整 CUDA/NCCL 包安装；`.buildkite/*.yaml`、`requirements/*.txt`（cuda/cpu/test/rocm-build/kv_connectors/xpu 等）、`pyproject.toml`、`docker/versions.json`、`.pre-commit-config.yaml` 同步把 torch-backend 由 cu129 改为 cu130、torchao 由 0.14.1 升到 0.17.0、causal-conv1d 由 1.5.2 升到 1.6.0。另有少量源码 API 适配：`csrc/cpu/utils.hpp` 用 `at::cpu::get_cpu_capabilities()` 的 `caps.at("l2_cache_size").toInt()` 替换 `at::cpu::L2_cache_size()`，`vllm/utils/torch_utils.py` 出现 `HAS_OPAQUE_TYPE = is_torch_equal_or_newer("2.12.0.dev")` 判断；还涉及 `vllm/distributed/parallel_state.py`、`vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`、`tests/kernels/helion/test_pattern_matching.py` 及一张 docs 图片。

## 关键改动

- `CMakeLists.txt`：torch 支持版本 2.10.0 → 2.11.0（CUDA 与 ROCm）。
- `docker/Dockerfile`、`Dockerfile.cpu`、`Dockerfile.s390x`、`versions.json`：CUDA_VERSION 12.9.1 → 13.0.0、基础镜像 ubuntu20.04 → 22.04。
- `requirements/*.txt` 与 `.buildkite/*.yaml`、`.pre-commit-config.yaml`：cu129 → cu130、torchao 0.14.1 → 0.17.0、causal-conv1d 1.5.2 → 1.6.0。
- `csrc/cpu/utils.hpp`：适配 torch 2.11 的 CPU L2 cache API。
- `vllm/utils/torch_utils.py`、`vllm/distributed/parallel_state.py`：torch 版本相关兼容判断。

```text
-set(TORCH_SUPPORTED_VERSION_CUDA "2.10.0")
-set(TORCH_SUPPORTED_VERSION_ROCM "2.10.0")
+set(TORCH_SUPPORTED_VERSION_CUDA "2.11.0")
+set(TORCH_SUPPORTED_VERSION_ROCM "2.11.0")
```

## 性能声明

PR 正文未给出定量性能数据（正文仅一句 "Update to Torch 2.11"，无 benchmark 数字、测试条件或加速比；regex 未检出 perf claims）。本卡所述版本与适配信息均来自 diff 素材，不构成性能结论。

## 验证与回归风险

验证：PR 正文未注明测试计划或测试结果；diff 中可见的 Buildkite CI 配置（`.buildkite/test-amd.yaml`、`test_areas/*.yaml`）、`tests/standalone_tests/python_only_compile.sh`、`tests/kernels/helion/test_pattern_matching.py` 改动本身即为构建/测试路径。可能受影响路径：CUDA 13.0 / cu130 与 torch 2.11、torchao 0.17.0、CPU / ROCm / XPU / s390x 多平台构建、CPU L2 cache API 变更、torch 2.12 opaque type 兼容判断；正文未列出回归测试与风险说明。

## 关联

PR 正文未给出同分类其它 PR 编号、前序或后续 PR，无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
