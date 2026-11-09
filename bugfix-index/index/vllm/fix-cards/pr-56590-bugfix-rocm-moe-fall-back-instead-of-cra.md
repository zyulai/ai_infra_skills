# PR #56590: [Bugfix][ROCm][MoE] Fall back instead of crashing when AITER MoE is requested for a non-gated (is_act_and_mul=False) model

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56590](https://github.com/vllm-project/vllm/pull/56590) |
| 作者 | shantipriya-amd |
| 合并日期 | 2026-09-17 |
| labels | bug, rocm, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
AITER MoE 请求 non-gated 模型时引擎初始化崩溃;改为告警并按优先级回退其余 MoE 后端。

## 问题现象
uplift 调查 `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` 时发现:设置 `VLLM_ROCM_USE_AITER_MOE=1` 后,任何非 gated 激活(`is_act_and_mul=False`)的 MoE 模型引擎初始化即崩溃。在 MI350X 实机(TP=8)上复现并验证修复。修复后回退到 Triton:GSM8K(5-shot)与基线(AITER_MOE=0)各指标在 1 个 stderr 内;`vllm bench serve` 并发参考(128 并发)显示回退路径不比默认更慢(如 1024/1024 吞吐 2.72→2.95 req/s),该表仅为确认无回归,非性能改进声明——此前 `AITER_MOE=1` 配置从未产生过可用运行。

## 根因
正文披露:`AiterExperts` 没有 non-gated activation(`is_act_and_mul=False`)对应的内核,而 `select_unquantized_moe_backend()` 在这种情形直接抛 `ValueError`,而不是回退到其他后端。

## 修复方案
`vllm/model_executor/layers/fused_moe/oracle/unquantized.py`:环境变量 opt-in 选中 AITER 后,不再直接 `_return_or_raise`,而是逐内核调用 `is_supported_config`;不支持时打明确说明回退的告警,把 AITER 从 `AVAILABLE_BACKENDS` 移除,继续按优先级尝试剩余后端:
```python
logger.warning_once(
    "VLLM_ROCM_USE_AITER_MOE=1 was requested, but %s "
    "Falling back to try the remaining available MoE backends: %s.",
    _make_log_unsupported(backend, reason),
    ", ".join(b.value for b in AVAILABLE_BACKENDS
              if b != UnquantizedMoeBackend.AITER))
if UnquantizedMoeBackend.AITER in AVAILABLE_BACKENDS:
    AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.AITER)
```
显式 `--moe-backend aiter` 对 non-gated 仍 raise(新增测试 `test_explicit_aiter_backend_non_gated_activation_still_raises` 断言);另增 fallback 行为测试。单元测试 24 passed, 3 skipped。

## 影响与回归风险
- 回归面:ROCm unquantized MoE 后端选择的 env-var opt-in 路径;AITER 不支持配置时多一条告警并回退,显式 pin、非 ROCm 及其他后端选择行为不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
