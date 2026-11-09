# PR #16458: Fix external_models import path and migrate model loading tests

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16458](https://github.com/sgl-project/sglang/pull/16458) |
| 作者 | alisonshao |
| 合并日期 | 2026-01-09 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 30 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
AMD CI 上 `external_models` 包导致 `ModuleNotFoundError`;把该包移入 `python/sglang/test/external_models/`,并把模型加载测试迁移到 `test/registered/model_loading/`。

## 问题现象
AMD CI 出现 `ModuleNotFoundError`(正文 Summary 一句话给出)。触发点与 `external_models` 的导入路径有关;正文未提供完整报错栈。

## 根因
正文未详述机制,只说明修法是「moving external_models to `python/sglang/test/external_models/`」。(由 diff 推断)`external_models` 原先位于 `test/srt/external_models/`,测试通过 `envs.SGLANG_EXTERNAL_MODEL_PACKAGE` / `envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE` 以顶层包名 `external_models` 引用;在 AMD CI 的运行环境下该路径不在可导入范围内,import 失败。迁入 `python/sglang/test/` 后按 `sglang.test.external_models` 引用即可正常导入。

## 修复方案
- 将 `test/srt/external_models/`(含 `custom_qwen2_vl.py` 等)整体移到 `python/sglang/test/external_models/`。
- 测试迁移并注册 CI:`test/srt/test_external_models.py` → `test/registered/model_loading/test_external_models.py`,其中包名改为:
```diff
-        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("external_models")
-        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("external_models")
+        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
+        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
```
并加 `register_cuda_ci` / `register_amd_ci`;`test_modelopt_export.py`、`test_modelopt_loader.py`、`test_utils_update_weights.py` 同样迁入 `test/registered/model_loading/` 并注册 `stage-b-test-small-1-gpu`。
- `test/srt/run_suite.py` 从 `per-commit-1-gpu` 及 AMD 套件中移除上述 TestFile 条目。
- `.github/workflows/pr-test.yml`:`stage-b-test-small-1-gpu` 分片从 11 扩到 14(`--auto-partition-size 14`)。
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`:`_cast_to_e8m0_with_rounding_up` 的装饰器改为 `@torch.compile(disable=_is_hip or _is_npu)`(diff 可见,正文未说明)。

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围。
- 回归面:测试目录与 CI 编排为主。任何仍以旧包名 `external_models` 引用这两个 env 变量的调用都会失败,需改用 `sglang.test.external_models`;`per-commit` 套件不再直接包含这些测试(改由 `test/registered/model_loading/` 注册路径执行);stage-b 分片数变化影响 CI 排布;deep_gemm 的 e8m0 cast 在 HIP/NPU 上不再走 torch.compile。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
