# PR #52545: [Bugfix][CI/Build] Fail closed when selected precompiled CUDA variant is unavailable

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52545](https://github.com/vllm-project/vllm/pull/52545) |
| 作者 | jungjiyu |
| 合并日期 | 2026-08-28 |
| labels | bug, documentation, ready, ci/build, nvidia |
| 热度 | 👍 2 · 💬 38 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论

VLLM_USE_PRECOMPILED 选中 CUDA 变体但元数据拉取失败时会静默回退到默认变体,装上 CUDA 不兼容的 wheel;改为直接抛错(fail closed)。

## 问题现象

以 `VLLM_USE_PRECOMPILED` 安装并选中某个 CUDA wheel 变体时,若该变体的元数据不可用,现有代码会回退到 root/default wheel 的元数据,且不校验 CUDA 兼容性。复现案例(#52544):PyTorch CUDA 12.9 选中 `cu129`,回退到的 wheel 却要求 `libcudart.so.13`,装出 CUDA 不兼容的环境。

## 根因

正文披露:`setup.py` 的 `determine_wheel_url()` 中存在 CUDA 变体 → root/default 元数据的回退逻辑——选中变体的 metadata 查找失败后,`try_default=True` 再去拉默认变体元数据。而当前的元数据与 wheel 选择逻辑没有任何手段验证 root/default 产物与所选 CUDA 变体兼容,回退等于在未验证的情况下替换用户选定的变体。正文给出修复前后流程对照:before 是「selected CUDA variant → metadata lookup fails → retry root/default metadata」,after 是「→ propagate the lookup failure」。

## 修复方案

- `setup.py` `determine_wheel_url()`:删除 `try_default` 回退——`fetch_metadata_for_variant(commit, variant)` 失败时不再 warning 后重试 `fetch_metadata_for_variant(commit, None)`,改为直接抛 `RuntimeError`,错误信息说明 root/default 变体因无法验证 CUDA 兼容性而不作回退,并给出两条出路:用 `VLLM_PRECOMPILED_WHEEL_LOCATION` 提供兼容 wheel,或关闭 `VLLM_USE_PRECOMPILED` 改为源码构建。
- 更新 `docs/contributing/ci/nightly_builds.md` 描述新的 CUDA 解析行为(显式 `VLLM_MAIN_CUDA_VERSION` 选变体,未设则从 PyTorch/nvidia-smi 检测,检测失败用默认值;CUDA 只拉取所选变体的元数据)。
- 为显式与自动检测的 CUDA 变体新增回归测试,并接入 Python-only Installation CI。回归测试在未修改的 base commit 上运行会失败(`Failed: DID NOT RAISE RuntimeError`,且会多调一次 `fetch(commit, None)`),补丁后 2 passed。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:仅影响 `VLLM_USE_PRECOMPILED=1` 的安装路径。行为从「静默装默认变体」变为「安装失败并报错」,对于此前依赖回退才能装上的场景(如某 CUDA 变体 nightly 元数据暂缺)是破坏性变化,需用户显式提供 wheel 或改源码构建;base commit 选择逻辑不变,ROCm precompiled wheel 路径不变,非 CUDA 安装不受影响。

## 关联

- #52544(修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
