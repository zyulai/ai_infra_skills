# PR #2611: Fix triton kernel performance regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2611](https://github.com/sgl-project/sglang/pull/2611) |
| 作者 | kkHuang-amd |
| 合并日期 | 2024-12-27 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
ROCm 上 triton 3.1.0 内核性能回退;ROCm 镜像改为源码安装 triton-3.0.0(commit 845d75a)绕过。

## 问题现象
正文描述:ROCm 平台的 triton kernel 在 triton-3.1.0 版本上出现性能下降(performance drop);作者未附具体数据、复现脚本或受影响内核清单。

## 根因
正文未披露性能回退的具体根因(未说明是 triton 哪个改动导致),只给出规避策略:强制回退使用 triton-3.0.0 版本作为 work-around。

## 修复方案
改 `docker/Dockerfile.rocm`:新增 `TRITON_REPO` / `TRITON_COMMIT`(845d75a)构建参数,`pip uninstall -y triton` 后从 triton-lang/triton 源码 checkout 指定 commit 并 `python3 setup.py install`,把 ROCm 镜像内的 triton 固定到回退版本(正文称 triton-3.0.0)。

```diff
+ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
+ARG TRITON_COMMIT="845d75a"
+
+RUN pip uninstall -y triton
+RUN git clone ${TRITON_REPO} \
+    && cd triton \
+    && git checkout ${TRITON_COMMIT} \
+    && cd python \
+    && python3 setup.py install
```

## 影响与回归风险
- 回归面:只影响 `docker/Dockerfile.rocm` 构建的 ROCm 镜像:triton 由 pip 预装版本改为源码构建的固定 commit;构建流程需访问外网 git 仓库、构建时间增加;CUDA / 非 ROCm 路径不受影响。在镜像外自行安装新版 triton 的用户不在本修复覆盖范围内。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
