# PR #17256: fix(ci): recover from corrupted MMMU parquet cache

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17256](https://github.com/sgl-project/sglang/pull/17256) |
| 作者 | harvenstar |
| 合并日期 | 2026-01-18 |
| labels | Multi-modal, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
HF 缓存中 MMMU parquet 损坏(ArrowInvalid: Parquet magic bytes not found)令 lmms_eval 失败且不产 JSON,MMM U 评测 CI 间歇性挂;检测到该错误后清缓存并以 force_redownload 重试一次。

## 问题现象
正文披露:当 HuggingFace 缓存中的 MMMU 数据集 parquet 文件损坏时报 `ArrowInvalid: Parquet magic bytes not found`,lmms_eval 进程直接失败且不产出 JSON 结果文件,导致 MMMU 评测 CI 测试失败(间歇性 CI failure)。正文附两个失败 run 链接(run 21076101101/job 60658633322 @pr=16388、run 21075841576/job 60653165682 @pr=16561)。影响面是所有依赖缓存跑 MMMU 评测的 CI job。

## 根因
正文披露:CI 机器的 HuggingFace 缓存里 MMMU parquet 文件损坏后,数据集加载库抛 ArrowInvalid,lmms_eval 子进程异常退出、不生成评测 JSON,CI 断言失败;此前没有任何从损坏缓存恢复的机制,只能人工干预。

## 修复方案
改 `python/sglang/test/kits/mmmu_vlm_kit.py`,新增自动恢复逻辑并把 `run_mmmu_eval` 中两处 `subprocess.run(cmd, check=True, timeout=3600)` 替换为 `_run_lmms_eval_with_retry(cmd)`:
1. `_is_mmmu_parquet_corruption`:仅当错误输出同时含 `ArrowInvalid`、`Parquet magic bytes not found` 且出现 `MMMU`/`lmms-lab--MMMU` 时判定为本 case;
2. `_cleanup_mmmu_dataset_cache`:定位并 `shutil.rmtree` 损坏的 `datasets--lmms-lab--MMMU` 缓存目录,优先 CI 约定路径 `/hf_home`,回退 `HF_HOME`(默认 `~/.cache/huggingface`);
3. 清理成功后以 `temp_set_env(HF_HUB_OFFLINE="0", HF_DATASETS_DOWNLOAD_MODE="force_redownload")` 重试一次;清理失败或其他错误照常抛出并打印 stdout/stderr。关键 diff 摘录:

```diff
+        if _is_mmmu_parquet_corruption(error_output):
+            print("Detected MMMU parquet corruption error. Attempting recovery...")
+            if _cleanup_mmmu_dataset_cache():
+                with temp_set_env(
+                    HF_HUB_OFFLINE="0",
+                    HF_DATASETS_DOWNLOAD_MODE="force_redownload",
+                ):
+                    subprocess.run(cmd, check=True, timeout=timeout)
```

## 影响与回归风险
- 回归面:仅测试基建(`mmmu_vlm_kit.py`)。错误识别条件收得很窄(三类字符串同时命中),非 MMMU parquet 损坏的失败走原有 raise 路径;行为差异是失败子进程输出改为先 capture 再打印(成功时同样回显),以及可能多一次整跑重试(最多重试一次)。风险点:重试会重复消耗最多 3600s 超时时长;`/hf_home` 优先级假设只在 Docker CI 环境成立。

## 关联
- https://github.com/sgl-project/sglang/actions/runs/21076101101/job/60658633322?pr=16388
- https://github.com/sgl-project/sglang/actions/runs/21075841576/job/60653165682?pr=16561

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
