# PR #14465: Fix safetensors validation to catch corruption after download

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14465](https://github.com/sgl-project/sglang/pull/14465) |
| 作者 | alisonshao |
| 合并日期 | 2025-12-06 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
safetensors 校验被 CI 环境变量门控且只查缓存、不查新下载文件,损坏权重加载时才报错;改为始终校验缓存并在下载后校验、清理损坏文件。

## 问题现象
权重下载损坏(truncated downloads、network errors 等)后,直到模型加载才以晦涩报错失败。CI 中的表现为:
```
safetensors_rust.SafetensorError: Error while deserializing header:
invalid JSON in header: EOF while parsing a value at line 1 column 0
```
关联 CI 失败:runs/19948236909/job/57203231359。

## 根因
正文披露:safetensors 校验只在 `SGLANG_IS_IN_CI=true` 设置时运行,但该环境变量在 GPU workflows(pr-test.yml、nightly-test-nvidia.yml)中缺失;此外校验只检查本地缓存文件,新下载的文件从不被校验。

## 修复方案
`python/sglang/srt/model_loader/weight_utils.py`:
1. `download_weights_from_hf` 移除 `is_in_ci()` 门控——总是先通过 `find_local_hf_snapshot_dir` 查找并校验本地缓存(校验会清理损坏文件),命中则跳过下载;
2. 新增 `_validate_weights_after_download()`:按 `allow_patterns` 收集下载的权重文件,对 `.safetensors` 逐个调用 `_validate_safetensors_file`,发现损坏则调用 `_cleanup_corrupted_files_selective` 删除损坏文件并抛出 `RuntimeError` 提示重试下载;
3. 下载完成后调用该校验。
`.github/workflows/pr-test.yml` 与 `.github/workflows/nightly-test-nvidia.yml` 顶层补 `env: SGLANG_IS_IN_CI: true`。

```diff
+    # Validate downloaded files to catch corruption early
+    _validate_weights_after_download(hf_folder, allow_patterns, model_name_or_path)
+
     return hf_folder
```

## 影响与回归风险
- 影响窗口:正文未说明;影响所有通过 `download_weights_from_hf` 下载权重的路径(CI 与本地)。
- 回归面:权重下载流程——本地缓存查找从 CI-only 变为总是执行(CI 之外也会校验缓存,可能带来额外 IO/启动开销);下载后新增逐文件 safetensors 反序列化校验,大模型下载后校验耗时增加;损坏文件被自动删除后抛错,行为从「加载时晦涩报错」变为「下载后立即报错并可重试」。

## 关联
- https://github.com/sgl-project/sglang/actions/runs/19948236909/job/57203231359 (相关 CI 失败)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
