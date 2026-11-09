# PR #59700: [Bugfix][CI] Widen DBO+DP+EP GSM8K accuracy margin on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59700](https://github.com/vllm-project/vllm/pull/59700) |
| 作者 | divakar-amd |
| 合并日期 | 2026-10-01 |
| labels | bug, rocm, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 DBO+DP+EP GSM8K 准确率 0.617 两次低于 0.62 阈值属采样噪声 flake;仅 ROCm 把 MIN_ACCURACY 放宽到 0.61。

## 问题现象
`tests/v1/distributed/test_dbo.py::test_dbo_dp_ep_gsm8k[deepep_high_throughput]` 在 amd-ci nightly 上失败:90 天内至少两次、分别发生在两种 ROCm 架构(MI300 与 MI355)上,两次报错信息完全相同:`AssertionError: DBO+DP+EP accuracy too low (deepep_high_throughput): 0.617 < 0.620`(附 MI300、MI355 两个 Buildkite 构建链接)。属间歇性 flake 而非确定性失败。

## 根因
正文披露:该 eval 以 `temperature=0.3`(取自模型 `generation_config.json`)、仅 256 道 GSM8K 题、且没有固定的 per-request seed 运行;在这个样本量下,低于 1 个百分点的偏差落在预期采样噪声范围内,并不必然表示真实回归。

## 修复方案
改动 `tests/v1/distributed/test_dbo.py`:把 `MIN_ACCURACY` 按平台区分,仅在 ROCm 上从 `0.62` 放宽到 `0.61`,CUDA 及其他平台保持不变。关键 diff 摘录:

```python
# Expected 0.64 with 2% buffer (based on vLLM test data)
# On ROCm, widened to 0.61 after observing 0.617 amd-ci nightly flake twice.
MIN_ACCURACY = 0.61 if current_platform.is_rocm() else 0.62
```

同时新增 `from vllm.platforms import current_platform` 导入;原 `MIN_ACCURACY = 0.62` 行的注释保留为说明(期望值 0.64 加 2% 缓冲)。

## 影响与回归风险
- 影响窗口:正文未说明(针对近 90 天 nightly 上两次观察到 0.617)。
- 回归面:仅 CI 测试门槛——ROCm 上该用例的通过阈值放宽 1 个百分点,介于 0.61-0.62 之间的真实小幅准确率回归可能被放过;CUDA/其他平台阈值不变;NUM_QUESTIONS(256)、NUM_SHOTS(5)等配置不动。

## 关联
- MI300 失败构建:https://buildkite.com/vllm/amd-ci/builds/13015/summary?jid=01a0a973-18d1-4f3f-925a-d78c307267cb&tab=output
- MI355 失败构建:https://buildkite.com/vllm/amd-ci/builds/13954/summary?jid=01a0f6b2-7bd8-486c-b337-f85dfee4cf7d&tab=output

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
