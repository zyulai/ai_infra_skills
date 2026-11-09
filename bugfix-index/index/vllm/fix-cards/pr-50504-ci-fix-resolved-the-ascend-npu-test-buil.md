# PR #50504: [CI][Fix] Resolved the Ascend NPU test build image fail and add file dependencies

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50504](https://github.com/vllm-project/vllm/pull/50504) |
| 作者 | yzeyu71 |
| 合并日期 | 2026-09-02 |
| labels | ready, ci/build |
| 热度 | 👍 2 · 💬 32 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
Ascend NPU CI 镜像构建失败(`triton-ascend==3.2.1` 在默认 PyPI 缓存找不到):给 uv 配置华为云 Ascend PyPI 镜像源,并加 `source_file_dependencies` 与 uv 加速。

## 问题现象
Ascend NPU 硬件测试 CI 在 Docker 镜像构建阶段失败:执行 `pip install -r /workspace/vllm-ascend/requirements.txt`(Dockerfile:30)时报
```
ERROR: No matching distribution found for triton-ascend==3.2.1
ERROR: failed to solve: process ... did not complete successfully: exit code: 1
```
原因上下文:构建期只配了集群内部 PyPI 缓存源(`cache-service-vllm...`),`triton-ascend` 不在默认 PyPI 缓存中。影响面:Ascend NPU 测试流水线整体无法构建镜像、CI 红灯。验证:本地镜像构建成功、CI 执行通过。

## 根因
正文披露:`triton-ascend` 包不在默认 PyPI 缓存源中,镜像构建阶段从 `vllm-ascend/requirements.txt` 安装依赖时找不到该包而失败。正文同时指出构建/测试用例耗时过长、打印大量日志不利于定位问题(作为第三项改动的动机)。镜像构建命令当时只在后续步骤通过 `PIP_EXTRA_INDEX_URL` 使用华为云 Ascend 源,而依赖预装步骤(`RUN pip install -r /workspace/vllm-ascend/requirements.txt`)没有带上该源。

## 修复方案
- `.buildkite/scripts/hardware_ci/run-npu-test.sh`(内嵌 Dockerfile):
  - 安装 uv 并配置镜像源:`UV_INDEX_URL` 指向内部 PyPI 缓存,`UV_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi`(托管 triton-ascend 的源);
  - 全部 `pip install` 换成 `uv pip install --quiet`,并在安装 vllm-ascend 依赖处显式加 `--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi` 与 `--index-strategy unsafe-best-match`:
    ```dockerfile
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv pip install -r /workspace/vllm-ascend/requirements.txt --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi --quiet --index-strategy unsafe-best-match
    ```
  - 缓存挂载点从 `/root/.cache/pip` 改为 `/root/.cache/uv`;安装 vLLM 时改用 `uv pip install -e` 并去掉 `-v`。
- `.buildkite/hardware_tests/ascend_npu.yaml`:新增 `source_file_dependencies` 参数(vllm/ 与 run-npu-test.sh,再用大量 `!` 排除 CUDA/XPU/TPU/CPU 专用路径、tests/docs 等),删除 `timeout_in_minutes: 20`,减少无关触发与日志。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围(报错时间戳为 2026-08-03 的 CI 日志)。
- 回归面:仅 Ascend NPU CI 流水线:镜像构建改用 uv(解析策略 `unsafe-best-match` 可能选中与 pip 不同的包版本组合);华为云 Ascend 源成为 extra index;`source_file_dependencies` 白名单会改变该流水线的触发条件——被排除路径的改动不再触发 NPU 测试,若排除范围过宽可能漏跑受影响的测试;其他硬件流水线与运行时行为不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
