# PR #49299: [Misc][Docs] Fix XPU compute-runtime driver link version mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49299](https://github.com/vllm-project/vllm/pull/49299) |
| 作者 | oonyshch |
| 合并日期 | 2026-07-21 |
| labels | documentation, intel-gpu, ready |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
XPU 安装指南中 "compute runtime 26.18" 链接指向 26.14.37833.4 旧版本,改为与 Dockerfile 一致的 26.18.38308.1。

## 问题现象
XPU 安装指南(`docs/getting_started/installation/gpu.xpu.inc.md`)中,链接文字 "compute runtime 26.18" 实际指向 release tag `26.14.37833.4`——既与链接文字不符,也与 `docker/Dockerfile.xpu` 中实际安装的驱动版本(`26.18.38308.1`)不一致;用户照文档安装会装到非推荐版本,可能遇到正文所述的 compatibility issue。

## 根因
正文披露:文档超链接指向的 compute-runtime 发布版本写错(维护中未与 Dockerfile.xpu 实际安装版本同步),属于文档链接版本错配,无代码影响。

## 修复方案
仅改 `docs/getting_started/installation/gpu.xpu.inc.md` 一行,把 "compute runtime 26.18" 的链接目标由 `https://github.com/intel/compute-runtime/releases/tag/26.14.37833.4` 改为 `https://github.com/intel/compute-runtime/releases/tag/26.18.38308.1`,使推荐安装版本与 Dockerfile.xpu 实际安装版本一致:
```diff
-- Start from vllm-xpu-kernels v0.1.10, we recommend user upgrade driver to [compute runtime 26.18](https://github.com/intel/compute-runtime/releases/tag/26.14.37833.4) release, to avoid potential compatibility issue.
+- Start from vllm-xpu-kernels v0.1.10, we recommend user upgrade driver to [compute runtime 26.18](https://github.com/intel/compute-runtime/releases/tag/26.18.38308.1) release, to avoid potential compatibility issue.
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:纯文档改动("Docs-only change"),正文验证了链接目标与 docker/Dockerfile.xpu 安装的 compute-runtime 版本一致;无任何代码/运行时影响,唯一依赖是文档读者按新链接安装推荐驱动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
