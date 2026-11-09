# PR #12135: [Bugfix] Revert PR #11435: Fix issues in CPU build Dockerfile. Fixes #9182

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#12135](https://github.com/vllm-project/vllm/pull/12135) |
| 作者 | terrytangyuan |
| 合并日期 | 2025-01-17 |
| labels | ready, ci/build |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #11435 的 CPU Dockerfile/bind-mount 与版本号回退改动:正确配置 podman 后不再需要该 workaround。

## 问题现象
本 PR 是回滚,正文未描述新的运行时现象。背景是 #11435("Fix issues in CPU build Dockerfile. Fixes #9182")针对 CPU 镜像构建(与 podman/buildah 环境)做的一组改动被判定不再需要,正文指见 #11435 中的讨论,并给出依据:「This is no longer needed if we configure podman correctly」(ref: containers/buildah discussion #4184)。即原 #9182 的构建问题应通过正确配置 podman 解决,而非 PR 里的变通。

## 根因
正文披露的根因:#11435 引入的变通方式(改用 `COPY` requirements 文件、`get_vllm_version` 失败时回退版本号)是不必要的,把 podman 配置正确(buildah discussion #4184 所述)即可,故整体回滚。#11435 所规避的底层构建失败细节正文未展开;从 diff 看被回滚的两处改动分别是:Dockerfile.cpu 中 requirements 文件由 bind mount 改为 `COPY`,以及 `get_vllm_version` 加了 `except LookupError: version = "0.0.0"` 的兜底(由 diff 推断)。

## 修复方案
回滚 commit 409475a827795000301f0d27582befab0563888b,两个文件:
- `Dockerfile.cpu`:删除 `COPY requirements-build.txt/common/cpu` 的方式,恢复在 pip 的 `RUN` 上以 bind mount 挂载 requirements 文件:

```diff
-COPY requirements-build.txt requirements-build.txt
 ARG PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
 ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
 RUN --mount=type=cache,target=/root/.cache/pip \
+    --mount=type=bind,src=requirements-build.txt,target=requirements-build.txt \
     pip install --upgrade pip && \
```

- `setup.py`:`get_vllm_version` 删除 try/except LookupError 回退 `"0.0.0"` 的逻辑,恢复直接调用 `get_version(write_to="vllm/_version.py")`。

## 影响与回归风险
- 影响窗口:#11435 合入到本回滚(2025-01-17)之间的版本,期间 CPU Dockerfile 构建路径与版本号推导按 workaround 行为执行。
- 回归面:改动只影响 CPU 镜像构建(`Dockerfile.cpu`)与源码构建时的版本号推导(`setup.py`)。要求构建环境正确配置 podman(否则 #11435 所规避的构建问题可能复现,正文认为正确配置后不存在该问题);`get_vllm_version` 在无法取得版本时将直接抛 LookupError 而不再回退 "0.0.0",版本号取值恢复严格行为。

## 关联
- vllm-project/vllm#11435(被回滚的 PR)
- vllm-project/vllm#9182(标题所述关联 issue)
- https://github.com/containers/buildah/discussions/4184(podman 正确配置的参考)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
