# PR #29060: [Docker] Optimize Dockerfile: consolidate apt-get and reduce image size by ~200MB

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @princepride |
| 合并 | 2025-11-25 |
| 分类 | Other（tier: major） |
| Label | ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/29060 |

## 一句话结论

合并 apt-get 与清理使构建 apt 阶段快 4%（665→639 秒）、base 层数减 33%，镜像约省 200MB。

## 动机

本 PR 优化 Dockerfile 以减小镜像体积、提升构建效率，做法是合并 apt-get 命令并补上规范清理，正文标注 Fixes #28644。改动前 base stage 有 6 个 RUN 命令，GCC-10 的安装与 update-alternatives 分散在多个独立 RUN 中，apt-get install 未统一带 `--no-install-recommends`，且多数安装块后没有删除 `/var/lib/apt/lists`。减少层数、避免推荐包与包列表残留带来的体积浪费是本次改动的动机。

## 优化点

把 GCC-10/g++-10 安装与 update-alternatives 合并进 base stage 的单条 apt-get 命令，base stage 的 RUN 由 6 条减到 4 条；对全部 4 处 apt-get install 补齐 `--no-install-recommends`（覆盖率 100%）；在每个 apt-get 块末尾追加 `rm -rf /var/lib/apt/lists/*`，清理命令由 1 处增到 3 处；包列表改为一行一个便于审阅。dev 与 vLLM-base 两个 stage 也各自补上 `--no-install-recommends` 与清理。改动集中在 docker/Dockerfile 单文件，属构建层优化，正文明确不面向用户、不更新 release notes。

## 关键改动

```text
docker/Dockerfile: 合并 base stage 的 apt-get 层、把 GCC-10 安装并入同一 RUN、补 --no-install-recommends 与 rm -rf /var/lib/apt/lists/*，并同步 dev / vLLM-base stage
```
```text
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        libibverbs-dev \
    && rm -rf /var/lib/apt/lists/* \
```

## 性能声明

PR 正文 Test Result 的定量结果：Build Time（apt-get 阶段）Before 665 秒（11m 5s）、After 639 秒（10m 39s），26 秒更快、4% reduction；Docker Layers（base stage）6 → 4 个 RUN，减少 2 层、33% reduction；`--no-install-recommends` 覆盖 100%（4/4 条 apt-get install）；清理命令由 1 增到 3；GCC-10 安装由 3 个 RUN 合并为 1。镜像体积为估算：包列表清理约 150 MB、避免推荐包约 50-100 MB，合计约 200-250 MB，正文注明 Based on Docker best practices、非实测。测试条件为静态分析（git diff、grep apt-get / rm -rf / --no-install-recommends）与 `docker build --target base --no-cache`（CUDA_VERSION=12.9.1、PYTHON_VERSION=3.12）；PR 未注明推理侧的模型/硬件/精度/workload 条件。

## 验证与回归风险

验证为静态分析加 base stage 构建测试，正文 Verification 列出：所有 apt-get install 均使用 `--no-install-recommends`、所有 apt-get 块后均有 `rm -rf /var/lib/apt/lists/*`、GCC-10 已并入 base stage、包组织改为一行一个、Hadolint 校验通过无新增错误。回归风险：改动仅涉及 Docker 构建镜像（label ci/build），正文称属内部构建优化、非用户可见；镜像体积缩减为估算而非实测；收益体现在构建时间与层数，未给出推理侧数据。

## 关联

PR 正文标注 Fixes #28644（issue）；未给出其他同区域优化 PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
