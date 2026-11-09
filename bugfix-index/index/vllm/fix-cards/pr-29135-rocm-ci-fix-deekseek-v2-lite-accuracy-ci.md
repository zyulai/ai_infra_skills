# PR #29135: [Rocm][CI] Fix DeekSeek V2-Lite Accuracy CI

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29135](https://github.com/vllm-project/vllm/pull/29135) |
| 作者 | charlifu |
| 合并日期 | 2025-11-22 |
| labels | rocm, ready, ci/build |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm 不支持 deepEP 且 MOE padding 令 eplb 报错,CI 脚本改为按平台选后端并关闭 padding。

## 问题现象
DeepSeek V2-Lite 精度 CI(以及 qwen30b_a3b_fp8_block_ep 相关 EP/EPLB 定时集成测试)在 ROCm 环境下失败:脚本默认使用 `deepep_high_throughput` / `deepep_low_latency` 后端,而 ROCm 不支持 deepEP;同时 `VLLM_ROCM_MOE_PADDING` 默认开启(该特性仅适用于 MI300X),会导致 eplb 抛错。正文未附具体报错日志。

## 根因
正文披露两点:1) ROCm 平台不支持 deepEP,而 CI 脚本无条件以 deepEP 系后端启动;2) `VLLM_ROCM_MOE_PADDING` 只对 MI300X 有效,在相关环境下开启会导致 eplb 抛出错误。

## 修复方案
修改两个 BuildKite 定时集成测试脚本 `.buildkite/scripts/scheduled_integration_test/deepseek_v2_lite_ep_eplb.sh` 与 `qwen30b_a3b_fp8_block_ep.sh`:加入平台探测(`rocm-smi` 命令存在、`/opt/rocm` 目录存在或 `ROCM_PATH` 非空视为 ROCm),ROCm 平台改用 `allgather_reducescatter` 后端并 `export VLLM_ROCM_MOE_PADDING=0`,非 ROCm 平台维持原来的 deepEP 后端:

```diff
+if command -v rocm-smi &> /dev/null || [[ -d /opt/rocm ]] || [[ -n "${ROCM_PATH:-}" ]]; then
+  BACKENDS=("allgather_reducescatter")
+  export VLLM_ROCM_MOE_PADDING=0
+else
+  BACKENDS=("deepep_high_throughput" "deepep_low_latency")
+fi
```

注:正文表述为"把 `VLLM_ROCM_MOE_PADDING` 默认值改为 0",但 diff 实际是在 CI 脚本中对该变量显式 `export 0`,仓库运行时代码的默认值未见改动。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响这两个 CI 脚本;CUDA 平台行为不变。ROCm 平台的测试覆盖由 deepEP 系后端切换为 `allgather_reducescatter`,deepEP 在 ROCm 上的路径不再被该 CI 覆盖(本就不支持);平台探测依赖 `rocm-smi`/`/opt/rocm`/`ROCM_PATH` 的存在性,极端环境下可能误判。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
