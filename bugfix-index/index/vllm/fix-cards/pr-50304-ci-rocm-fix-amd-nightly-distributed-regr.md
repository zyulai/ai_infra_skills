# PR #50304: [CI][ROCm] Fix AMD nightly distributed regressions

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50304](https://github.com/vllm-project/vllm/pull/50304) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-07-29 |
| labels | rocm, ready, v1 |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
未落地的 all-gather API 令 AMD nightly 7 个 custom-all-reduce 测试失败、shutdown 测试卡 2 GiB 阈值;回退测试并调 ROCm 阈值。

## 问题现象
AMD nightly 两类失败:(1) Distributed Tests (4 GPUs):`test_custom_all_reduce.py` 中 7/7 custom-all-reduce 用例失败,报缺失 `custom_all_gather`;(2) Distributed Torchrun + Shutdown Tests (2 GPUs):teardown 显存降到 2.01 GiB,却仍对着严格的 2 GiB 阈值超时。依据为 buildkite 构建 11417 的两个失败链接。

## 根因
正文披露:一是 PR #50089 在对应的 `CustomAllreduce` Python API(尤其 custom all-gather/reduce-scatter)落地之前就添加了测试,这些 AG/RS 集成仍留在未合并的 Kimi K3 umbrella PR #50000 中且是 CUDA-only,不加平台门控直接拷贝到 ROCm 必然失败;二是 PR #45195 引入的 in-process shutdown 测试,被 #49270 在 MI300 上暴露出其严格的 2 GiB 阈值过紧——ROCm 实际只能到 4 GiB 的 idle-runtime 底线。

## 修复方案
- `tests/distributed/test_custom_all_reduce.py`:只回退 #50089 提前加入的 117 行 AG/RS 测试(删除 `test_sp16_dispatches_only_to_mnnvl_lamport`、`test_custom_collectives_world_size_four` 及 graph/eager 用例中的 all-gather/reduce-scatter 段),保留此前 nightly 已通过的 4 个 ROCm custom-all-reduce 用例。
- `tests/v1/shutdown/test_delete.py`:shutdown 测试在 ROCm 上启用 helper 既有的 4 GiB idle-runtime 底线并加 60 秒内层超时,CUDA 维持原有 2 GiB 阈值:
```diff
         wait_for_gpu_memory_to_clear(
             devices=[0],
             threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
+            threshold_ratio=0.01 if current_platform.is_rocm() else None,
+            timeout_s=(
+                SHUTDOWN_TEST_TIMEOUT_SEC // 2
+                if current_platform.is_rocm()
+                else SHUTDOWN_TEST_TIMEOUT_SEC
+            ),
         )
```

## 影响与回归风险
- 影响窗口:#50089 / #45195+#49270 合并之后的 AMD nightly CI。
- 回归面:纯测试改动,不触及生产代码。回退 AG/RS 测试后,`custom_all_gather`/`custom_reduce_scatter` 的 ROCm 覆盖暂时消失,待 #50000 落地后再恢复;shutdown 阈值放宽仅对 `current_platform.is_rocm()` 生效,CUDA 路径阈值与超时不变。

## 关联
- PR #50089(过早添加 AG/RS 测试)
- PR #50000(Kimi K3 umbrella,AG/RS 集成所在,未合并)
- PR #45195(引入 in-process shutdown 测试)
- PR #49270(在 MI300 上暴露 2 GiB 阈值问题)
- buildkite AMD CI 构建 11417 两个失败任务链接(见正文)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
