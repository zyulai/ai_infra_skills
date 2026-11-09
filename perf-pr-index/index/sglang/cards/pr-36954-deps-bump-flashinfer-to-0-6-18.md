# PR #36954: [Deps] Bump FlashInfer to 0.6.18

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-08-31 |
| 分类 | Attention / Kernel(classify.py 给出,tier: major) |
| Label | documentation, dependencies, run-ci, bypass-fastfail, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示约 9 个文件,含 3 个文件删除 |
| 链接 | https://github.com/sgl-project/sglang/pull/36954 |

## 一句话结论

将 FlashInfer 从 0.6.17 升级到正式版 0.6.18,并顺势删除已上游化的 Kimi-K3 专用 Docker/DeepEP 补丁构建链。

## 动机

FlashInfer 0.6.18 正式发布(此前 cu134 镜像一直用 0.6.18rc10 预发布版),SGLang 需要对齐 Docker 镜像、pyproject pin 与运行时版本断言。同时,Kimi-K3 专用镜像中维护的 DeepEP 运行时补丁(LL topk 16、SWITCH_HIDDEN 3584、SourceMeta 对齐、CUDA 13 cccl 修复等)已随上游解决,专用 Docker 构建链可整体删除,降低镜像维护成本。PR 正文除 CI 状态外无文字说明,以上动机来自 diff 内容本身。

## 优化点

依赖版本升级加构建链清理:`docker/Dockerfile` 的 `FLASHINFER_VERSION` 0.6.17→0.6.18;`docker/Dockerfile.cu134` 从 0.6.18rc10 改为正式 0.6.18;`pyproject.toml` 的 `flashinfer_python[cu13]==0.6.18`(保持与 Dockerfile jit-cache 版本对齐);`engine.py` 的 `assert_pkg_version("flashinfer_python", "0.6.18")` 运行时断言同步。删除 `docker/kimi_k3/` 下三个文件:`apply_deepep_k3_patch.sh`(169 行 DeepEP 源码补丁脚本)、`kimi_k3_cu12.Dockerfile`、`kimi_k3_cu13.Dockerfile`;Kimi-K3 cookbook 文档同步移除对 pinned 0.6.17 的描述。另有一个测试 docstring 更新(移除 pin 版本引用)。

## 关键改动

```text
docker/Dockerfile: FLASHINFER_VERSION 0.6.17 -> 0.6.18
docker/Dockerfile.cu134: 0.6.18rc10 -> 0.6.18 正式版
python/pyproject.toml: flashinfer_python[cu13]==0.6.18
python/sglang/srt/entrypoints/engine.py: flashinfer_python 版本断言改为 0.6.18
docker/kimi_k3/(3 个文件删除): Kimi-K3 专用镜像与 DeepEP 补丁脚本移除
```

```python
assert_pkg_version(
    "flashinfer_python",
    "0.6.18",
    "Please uninstall the old version and "
    "reinstall the latest version by following the instructions "
    "at https://docs.flashinfer.ai/installation.html.",
)
```

## 性能声明

PR 正文未提供 benchmark 或性能对比数据,未注明测试条件(0.6.18 的内核变化清单也未在正文列出)。

## 验证与回归风险

CI 快照:PR Test (Base) 与 (Extra) 通过,AMD ROCm 7.2 失败。风险点:运行时断言强制 0.6.18,旧环境需重装;cu134 镜像从 rc10 切正式版,rc 期间引入/修复的行为随之固化;删除 Kimi-K3 专用镜像意味着依赖该镜像的自定义部署(topk16 DeepEP、DeepGEMM 0.1.5.post2 SiTU、nvimgcodec 等)需改用官方镜像或自行构建;Kimi-K3 cookbook 中 MoE runner/attention backend 的自动选择描述随 pin 移除而更新。

## 关联

同区域 FlashInfer bump 链:#36954(0.6.18)→ 后续 #43494(0.7.1)、#44591(0.7.2);Kimi-K3 相关部署见 PR #44036(MegaMoE SiTU/NVFP4)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
