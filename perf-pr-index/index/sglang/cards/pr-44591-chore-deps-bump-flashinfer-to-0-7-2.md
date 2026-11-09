# PR #44591: [Chore] deps: Bump FlashInfer to 0.7.2

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-10-16 |
| 分类 | Attention / Kernel(classify.py 给出,tier: major) |
| Label | dependencies, run-ci, run-ci-extra, release-highlight, bypass-fail-fast, parallel-stages, max-concurrency |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 4 个文件,均为版本号改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/44591 |

## 一句话结论

将 FlashInfer 依赖从 0.7.1 升级到 0.7.2,让 SGLang 全量对齐 FlashInfer 新版本内核与修复。

## 动机

FlashInfer 上游发布 0.7.2,SGLang 需要保持 Docker 镜像、Python 依赖与运行时版本断言三方一致,以便用户拿到 FlashInfer 0.7.x 系列的内核更新与修复。PR 正文除一句话 Summary 外未给出更多动机说明(如具体修复清单或性能收益),本卡如实不引用。

## 优化点

纯依赖版本升级,不含 SGLang 侧算法或调度改动。四处改动保持一致:`docker/Dockerfile` 与 `docker/Dockerfile.cu134` 中 `FLASHINFER_VERSION` 构建参数从 0.7.1 改为 0.7.2;`python/pyproject.toml` 中 `flashinfer_python[cu13]==0.7.2`(注释注明需与 Dockerfile 的 jit-cache 版本对齐);`python/sglang/srt/entrypoints/engine.py` 中 `_set_envs_and_config` 的 `assert_pkg_version("flashinfer_python", "0.7.2", ...)` 运行时断言同步更新,旧版本会在启动时报错并提示按官方文档重装。

## 关键改动

```text
docker/Dockerfile: FLASHINFER_VERSION 构建参数 0.7.1 -> 0.7.2
docker/Dockerfile.cu134: 同上(CUDA 13.4 镜像,INSTALL_FLASHINFER_JIT_CACHE=1)
python/pyproject.toml: flashinfer_python[cu13] pin 0.7.1 -> 0.7.2
python/sglang/srt/entrypoints/engine.py: 启动时 flashinfer_python 版本断言改为 0.7.2
```

## 性能声明

PR 正文未提供任何 benchmark 或性能对比数据,未注明测试条件。

## 验证与回归风险

PR 正文附带的 CI 状态:PR Test (Extra) 通过,PR Test (Base) 失败,AMD ROCm 10 在快照时仍在排队;最终以合并时 CI 为准。风险点:运行时 `assert_pkg_version` 强制要求 0.7.2,未升级的旧环境启动即失败,需按提示重装;Dockerfile 注释要求 jit-cache 版本与 pip pin 对齐,自建镜像若只改一处会出现版本漂移。

## 关联

前序:FlashInfer 0.7.1 bump(PR #43494)、FlashInfer 0.6.18 bump(PR #36954)。受益方包括依赖 FlashInfer 的 MegaMoE/MLA/TRT-LLM MoE 路径(如 PR #44036)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
