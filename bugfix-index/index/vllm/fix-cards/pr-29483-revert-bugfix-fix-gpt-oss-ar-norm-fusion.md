# PR #29483: Revert "[Bugfix] Fix GPT-OSS AR+NORM fusion (#28841)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29483](https://github.com/vllm-project/vllm/pull/29483) |
| 作者 | hl475 |
| 合并日期 | 2025-11-26 |
| labels | ready, ci/build, gpt-oss |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
原 GPT-OSS AR+NORM 融合修复 PR 使 LoRA TP 分布式测试失败,整体 revert 恢复 MoE all-reduce 后再切片的旧行为。

## 问题现象
原 PR #28841 合并后,[LoRA TP Test (Distributed)](https://buildkite.com/vllm/ci/builds/40685/steps/table?jid=019abd08-ecfe-4c0a-aa74-338de6151884) buildkite job 失败;经 Slack 讨论,决定先把该 PR revert 掉。revert PR 本身无新增测试结果。

## 根因
正文披露 #28841 breaks LoRA TP Test (Distributed),但未说明具体破坏机制。由 diff 推断:被回退的改动曾在 `fused_moe/layer.py` 的 `reduce_output` 中把 fused 输出在 all_reduce 之前按 `og_hidden_states` 切片以启用融合;revert 恢复为 all_reduce 之后再切片,并把 `symm_mem.py` 的 `copy_(inp.reshape(-1))` 恢复为 `copy_(inp.view(-1))`。

## 修复方案
整体 revert commit 6330f9477db214477004df6546f86e3f14f8eab9:
- `vllm/model_executor/layers/fused_moe/layer.py`:删除 all_reduce 前的 `states = states[..., :og_hidden_states]` 切片,zero-expert 与 shared/fused 输出路径恢复为 reduce 后再切片;
- `vllm/distributed/device_communicators/symm_mem.py`:恢复 `self.buffer[: inp.numel()].copy_(inp.view(-1))`;
- 删除 `tests/compile/distributed/test_fusions_e2e.py` 中 openai/gpt-oss-20b 用例及 `.buildkite/test-pipeline.yaml` 中 `vllm/model_executor/layers/fused_moe/layer.py` 触发行。

## 影响与回归风险
- 影响窗口:正文未写版本范围。
- 回归面:MoE 层 TP/序列并行的输出切片顺序、symm_mem all_reduce 输入视图、编译融合 e2e 测试与 CI 触发规则;#28841 原本想修的 GPT-OSS 融合问题会随之重新出现。

## 关联
[#28841](https://github.com/vllm-project/vllm/pull/28841)、[buildkite build 40685](https://buildkite.com/vllm/ci/builds/40685/steps/table?jid=019abd08-ecfe-4c0a-aa74-338de6151884)、[Slack 讨论](https://vllm-dev.slack.com/archives/C07R5PAL2L9/p1764117810034919)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
