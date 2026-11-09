# PR #9260: Revert "[Misc] feat: Deepgemm update for sgl-kernel (#8790)" to fix kernel CI

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9260](https://github.com/sgl-project/sglang/pull/9260) |
| 作者 | hnyls2002 |
| 合并日期 | 2025-08-17 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
回滚 #8790 对 sgl-kernel 的 DeepGEMM 升级,恢复原有构建方式,修复 kernel CI 失败。

## 问题现象
合并 #8790 后 kernel CI 失败,PR 正文直接给出失败的 CI run 链接(https://github.com/sgl-project/sglang/actions/runs/17002005000/job/48209894546)。正文未描述除 CI 失败之外的具体现象,也未说明失败的具体报错内容。

## 根因
正文未披露根因,仅说明该 PR 是对 commit a3d99d6dcdddcfb6bde7b50c0830f0ad5534d74d(即 #8790)的整体回滚,并附上失败 CI 链接,同时指出「要重新引入 deepgemm 升级,依赖 #9167」。

从 diff 角度看改动所暗示的机制:#8790 曾把 DeepGEMM 从「按 CUDA 版本选择仓库/tag、仅安装 JIT 源码」改为「固定 sgl-project/DeepGEMM 的 cabi 分支 + 新增 fmt 依赖 + 编译 deep_gemm_cpp C++ 扩展模块(链接 nvrtc)并安装编译产物」。回滚恢复为按 CUDA 版本选择 DeepGEMM 仓库与 tag(CUDA 12.8/12.9 用 sgl-project/DeepGEMM 的 blackwell 分支,其他版本用 deepseek-ai/DeepGEMM 固定 commit),只安装 deep_gemm 源码目录供运行时 JIT,不再编译 C++ 模块。新构建路径(编译扩展、fmt/nvrtc 依赖)疑似是 CI 失败的直接来源(由 diff 推断)。

## 修复方案
整体 revert,仅改动 `sgl-kernel/CMakeLists.txt`:
- 删除 `repo-fmt` 的 FetchContent 声明。
- 恢复按 CUDA 版本选择 DeepGEMM 仓库与 tag 的逻辑:

```cmake
if("${CUDA_VERSION}" VERSION_EQUAL "12.8")
  set(DeepGEMM_REPO "https://github.com/sgl-project/DeepGEMM")
  set(DeepGEMM_TAG "blackwell")
...
  set(DeepGEMM_REPO "https://github.com/deepseek-ai/DeepGEMM")
  set(DeepGEMM_TAG "391755ada0ffefa9a6a52b6f14dcaf22d1a463e0")
```

- 删除 `deep_gemm_cpp` Python 模块的编译、链接(nvrtc、mscclpp_static)与安装逻辑,以及空 `__init__.py` 的生成。
- 恢复仅安装 `deep_gemm/` 源码目录(排除 `.git*`、`__pycache__`)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:影响 sgl-kernel 的构建配置,即 DeepGEMM 相关的依赖拉取、编译与安装路径;使用 sgl-kernel 预编译包或从源码构建 DeepGEMM 集成的用户会回到 #8790 之前的构建行为。deepgemm 升级需等 #9167 落地后才可重新引入,短期内 DeepGEMM 停留在旧版本源码。

## 关联
- [#8790](https://github.com/sgl-project/sglang/pull/8790)(被回滚的 DeepGEMM 升级)
- [CI 失败 run](https://github.com/sgl-project/sglang/actions/runs/17002005000/job/48209894546)
- [#9167](https://github.com/sgl-project/sglang/pull/9167)(重新引入 deepgemm 升级的前提)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
