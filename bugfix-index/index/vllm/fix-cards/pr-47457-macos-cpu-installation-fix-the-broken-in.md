# PR #47457: [macOS][CPU][Installation] Fix the broken installation of vllm 0.24.0 in macos + cpu

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47457](https://github.com/vllm-project/vllm/pull/47457) |
| 作者 | WindChimeRan |
| 合并日期 | 2026-07-06 |
| labels | ready, ci/build, nvidia |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
tokenspeed-mla 依赖无平台 marker 致 macOS 解析安装 vllm 0.24.0 直接失败,加 `platform_system == "Linux"` marker 修复。

## 问题现象
macOS(arm64/M5 Pro)上执行 `uv pip install vllm==0.24.0` 在依赖解析阶段即失败:`tokenspeed-mla==0.1.2` 没有匹配平台 tag(如 `macosx_26_0_arm64`)的 wheel;单独安装 tokenspeed-mla 同样失败,该包只有 `manylinux_2_28_aarch64`、`manylinux_2_28_x86_64` wheel。失败发生在任何构建期设备检测之前,macOS 上完全装不上 vllm 0.24.0。

## 根因
正文披露:`requirements/cuda.txt` 中 `tokenspeed-mla==0.1.2` 没有 platform marker,而该包本身只发布 manylinux wheel(无 sdist);`setup.py` 的 `get_requirements()` 把该文件的行原样拷贝进 vLLM 发布的 `install_requires`,于是这条无条件依赖随每个发布版本带到所有平台,macOS 解析必败。

## 修复方案
在 `requirements/cuda.txt` 给该依赖加 Linux 平台 marker:

```diff
-tokenspeed-mla==0.1.2
+tokenspeed-mla==0.1.2; platform_system == "Linux"
```

正文验证了 marker 机制:platform marker 由安装器在执行 `pip install` 的机器上求值;在 Darwin 上该 marker 求值为 False,Linux 上为 True,并确认 `setup.py` 的 `_read_requirements()` 解析会原样保留 marker。验证日志中加 marker 后 `uv pip install` 通过(Checked 1 package)。

## 影响与回归风险
- 影响窗口:正文提到 vllm 0.24.0 安装受损,未给出更早的引入版本。
- 回归面:仅 `requirements/cuda.txt` 一行;Linux 安装路径不受影响(marker 在 Linux 为 True),非 Linux 平台不再安装该依赖,这正是修复目标。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
