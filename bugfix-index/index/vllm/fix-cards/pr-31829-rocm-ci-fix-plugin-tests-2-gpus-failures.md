# PR #31829: [ROCm][CI] Fix plugin tests (2 GPUs) failures on ROCm and removing `VLLM_FLOAT32_MATMUL_PRECISION` from all ROCm tests

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31829](https://github.com/vllm-project/vllm/pull/31829) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-01-07 |
| labels | rocm, ready, ci/build |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

ROCm CI 三个测试组失败:卸掉临时 tf32 环境变量,并把 albumentations 钉到 1.4.6。

## 问题现象

ROCm CI 中三个测试组失败:`plugin tests (2 GPUs)`、`Multi-Modal Models Test (Standard)`、`Entrypoints Integration Test (API Server 1)`。其中插件测试的失败源于 plugin 测试过程中自动安装了更新版本的 albumentations,与 tensorV2 的 albumentations 函数不兼容;正文未给出具体报错日志。

## 根因

按正文披露有两点:

1. 这些测试组此前带着 `VLLM_FLOAT32_MATMUL_PRECISION="tf32"` 环境变量运行;该设置属于临时规避手段。(由 diff 推断)被删除的注释写着 "Need tf32 to avoid conflicting precision issue with terratorch on ROCm. TODO: Remove after next torch update",说明它是为 terratorch 精度冲突打的补丁,约定 torch 更新后移除,现已到期。
2. plugin 测试会自动安装一个更新版本的 albumentations,导致与 tensorV2 albumentations 函数的兼容性问题,需要显式钉版。

## 修复方案

改动两个文件:

- `.buildkite/test-amd.yaml`:在 Entrypoints Integration Test (API Server 1)、Multi-Modal Models Test (Standard)、plugin tests 三处,把 `VLLM_FLOAT32_MATMUL_PRECISION="tf32" pytest ...` 命令还原为普通 `pytest ...`,分别对应 `entrypoints/openai/test_vision_embeds.py`、`models/multimodal/pooling/test_prithvi_mae.py`、`plugins_tests/test_io_processor_plugins.py`,并删除相应注释。
- `requirements/rocm-test.txt`:新增 `albumentations==1.4.6`(注释 "Required for plugins test"),阻止 plugin 测试自动装上不兼容的新版本。

关键 diff 摘录:

```diff
-  # Need tf32 to avoid conflicting precision issue with terratorch on ROCm.
-  # TODO: Remove after next torch update
-  - VLLM_FLOAT32_MATMUL_PRECISION="tf32" pytest -v -s entrypoints/openai/test_vision_embeds.py
+  - pytest -v -s entrypoints/openai/test_vision_embeds.py
```

```diff
+# Required for plugins test
+albumentations==1.4.6
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:仅 ROCm CI 配置(`.buildkite/test-amd.yaml`)与 ROCm 测试依赖(`requirements/rocm-test.txt`),不影响引擎运行时代码。移除 tf32 后,若 terratorch 相关精度冲突在当前 ROCm/torch 组合下复现,这三个测试可能再次失败;albumentations 钉到 1.4.6 后,依赖其新版本 API 的测试未来需手动升级。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
