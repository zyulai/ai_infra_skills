# PR #48154: [ROCm] Revert Part of `[ROCm] Fix pooling startup workspace lock` #47912

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48154](https://github.com/vllm-project/vllm/pull/48154) |
| 作者 | micah-wil |
| 合并日期 | 2026-07-09 |
| labels | rocm, ready, v1 |
| 热度 | 👍 3 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#47912 中 ROCm warmup 追加 max_num_reqs 的改动导致 AMD CI 失败,本 PR 回退该部分逻辑恢复原有 warmup 行为。

## 问题现象
#47912 合入后,AMD CI 出现若干失败,例如 Model Executor 中的:
`pytest -v -s tests/model_executor/model_loader/test_reload.py::test_reload_weights[inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC-inference-optimization/DeepSeek-V3-debug-multiply-FP8_DYNAMIC-inference-optimization/DeepSeek-V3-debug-add-FP8_DYNAMIC-1]`
(失败示例见 Buildkite 构建 77216 的日志,链接见正文。)

## 根因
正文披露:#47912("[ROCm] Fix pooling startup workspace lock")中的部分改动导致 AMD CI 失败,故回退该部分。正文未进一步解释失败的具体机制(由 diff 推断:被回退的代码在 ROCm、非 V2 ModelRunner 且启用 CUDA graph 时,会把 `max_num_reqs` 追加进 `warmup_sizes` 强制额外 warmup,这部分行为被整体移除)。

## 修复方案
回退 `vllm/v1/worker/gpu_worker.py` 中 `compile_or_warm_up_model` 的一部分:
1. 删除 `cg_capture_sizes` 在函数开头的提前计算,移回 `CompilationMode.VLLM_COMPILE` 分支内;
2. 整体删除「TODO(LucasWilkinson, akaratza): Remove when MRV1 is deprecated」标记的 ROCm MRV1 专属逻辑块(将 `max_num_reqs` 追加到 warmup_sizes 的代码)。

关键 diff 摘录(被删除的核心块):
```python
if (
    current_platform.is_rocm()
    and not self.use_v2_model_runner
    and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    and get_pp_group().is_last_rank
):
    ...
    if (max_num_reqs not in cg_capture_sizes
            and max_num_reqs not in warmup_sizes):
        warmup_sizes.append(max_num_reqs)
```

## 影响与回归风险
- 回归面:`gpu_worker.py` 的 compile/warmup 流程;ROCm MRV1 非 V2 ModelRunner 场景不再为 `max_num_reqs` 额外 warmup,#47912 原本要解决的 pooling 启动 workspace lock 问题在该路径上可能重新出现(正文未评估);CUDA/NVIDIA 路径行为不变。

## 关联
- [PR #47912](https://github.com/vllm-project/vllm/pull/47912)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
