# PR #22537: Add runai-model-streamer into Python packages installed in Dockerfile and fix NotADirectoryError Docker regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22537](https://github.com/sgl-project/sglang/pull/22537) |
| 作者 | amacaskill |
| 合并日期 | 2026-04-14 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
官方镜像补装 runai-model-streamer 的 s3/gcs/azure extras,并修复 #22160 使 kernels.lock 被 mv 成普通文件导致的 NotADirectoryError。

## 问题现象
两个问题:
1. 用官方 Docker 镜像以 `--load-format=runai_streamer` 从 Google Cloud Storage(`gs://...`)加载模型时,server 崩溃:`ImportError: GCS files module not found. Please install the required package.`(出自 runai_model_streamer/s3_utils/s3_utils.py:137)。
2. 含 #22160 的 nightly 镜像(`lmsysorg/sglang:nightly-dev-20260410-3aaaf53f`)启动即失败:`NotADirectoryError: [Errno 20] Not a directory: '/root/.cache/sglang/model_streamer'`(经 `server_args._maybe_download_model_for_runai` → `runai_utils.get_path` 的 `os.makedirs(...)` 抛出);`lmsysorg/sglang:v0.5.10` 与 `latest` 正常。

## 根因
正文披露两点:
1. runai-model-streamer 包近期把云厂商依赖拆分为 extras,镜像构建时未显式请求 GCS/Azure 依赖,故开箱不支持 GCS。
2. #22160 重构 Dockerfile 时加入 `&& mv python/kernels.lock /root/.cache/sglang`;该构建阶段 `/root/.cache/sglang` 目录尚不存在,mv 把 kernels.lock 改名成了名为 `sglang` 的普通文件而非移入目录。之后 Run:AI model streamer 初始化时试图在该路径下创建缓存目录(`os.makedirs(os.path.join(base_dir, "model_streamer"), exist_ok=True)`),因 `/root/.cache/sglang` 是普通文件,OS 抛出 `NotADirectoryError: [Errno 20]`。

## 修复方案
改 `docker/Dockerfile` 两处:
- 在 pip 安装清单中加入完整 extras:
  ```diff
  -    termplotlib
  +    termplotlib \
  +    "runai-model-streamer[s3,gcs,azure]>=0.15.7"
  ```
- 修复 mv 目标路径(先建目录再移入,并以 `/` 结尾):
  ```diff
  -    && mv python/kernels.lock /root/.cache/sglang \
  +    && mkdir -p /root/.cache/sglang \
  +    && mv python/kernels.lock /root/.cache/sglang/ \
  ```
  同时顺带调整了 kernels download 重试循环的缩进、删除了构建尾部的 `tests` 目录与 `*.pyc` 清理行。
注:pack 的正文在 20k 字符处截断(截断部分为 before/after 构建日志),diff 完整无截断。

## 影响与回归风险
- 影响窗口:正文未说明;按引入关系,runai 云依赖拆分影响近期镜像,GCS 路径问题自 extras 拆分起;NotADirectoryError 窗口为 #22160 合入后的 nightly 镜像。
- 回归面:仅 Docker 镜像构建(`docker/Dockerfile`);镜像内新增 runai extras 依赖体积略增;`/root/.cache/sglang` 恢复为真实目录,kernels.lock 正确落位;删除 `*.pyc`/tests 清理会略增镜像体积,不影响运行逻辑。

## 关联
- https://github.com/sgl-project/sglang/pull/22160 (引入 Docker 回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
