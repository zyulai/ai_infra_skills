# PR #43186: [CI] Lower granite-4.0-h-tiny gsm8k threshold for Hybrid SSM NixlConnector PD accuracy tests (4 GPUs)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43186](https://github.com/vllm-project/vllm/pull/43186) |
| 作者 | haosdent |
| 合并日期 | 2026-05-20 |
| labels | bug, kv-connector, nvidia, ready, v1 |
| 热度 | 👍 1 · 💬 9 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

#42430 使 granite-4.0-h-tiny 在 NIXL PD 下 gsm8k 分布变宽变低致 CI 失败;expected 降为 0.77、RTOL 放宽至 0.05。

## 问题现象

CI「Hybrid SSM NixlConnector PD accuracy tests (4 GPUs)」失败(失败构建 build 66835):`ibm-granite/granite-4.0-h-tiny` 的 gsm8k 从 ~0.80 变为更低且更宽的分布。近期 4 次观测:0.7498(#66759)、0.7415(#66782)、0.7559(#66835)、0.78(近期 rerun),跨度约 0.04。

## 根因

正文披露:#42430 把 D 侧 1-token recompute 从 `mamba_chunk_scan_combined_varlen` 改到 `selective_state_update`。作者用 fp64 参考、bf16 输入(granite 的 dtype)调研 kernel 分歧:两个 kernel 都到 bf16 ULP floor(~3.7e-3 vs fp64),但落点不同——chunk_scan 经 bf16 matmul 且有中间 bf16 cast,SSU 全程以 fp32 做 SSM step;两者无对错之分,只是产生相近误差的不同 bf16 输出。模型公布的 gsm8k 基线是在 chunk_scan 误差模式下测的,SSU 路径落入略不同的数值域,翻转少量 greedy-decode token。kernel 级分歧(`tl.dot` bf16 cast vs fp32 reduce)留待单独修改。

## 修复方案

`tests/v1/kv_connector/nixl_integration/test_accuracy.py`:granite-4.0-h-tiny 的 expected 0.80 → 0.77,RTOL 0.03 → 0.05,通过窗口 (0.72, 0.82) 覆盖全部 4 次观测并留约 2 个百分点余量;加 `TODO(#43186)` 注释指回本 PR,待 kernel 分歧修复后再收紧。

```diff
-RTOL = 0.03
+# TODO(#43186): Widened from 0.03 to absorb chunk_scan/SSU numeric jitter
+# on granite-4.0-h-tiny under NIXL PD; tighten when the kernel divergence
+# is fixed.
+RTOL = 0.05
```

## 影响与回归风险

- 影响窗口:#42430 合并之后(D 侧 recompute 走 selective_state_update 的场景)。
- 回归面:仅测试阈值文件 test_accuracy.py 的 RTOL 与 granite expected 值,其他模型 expected 值未动;不涉及生产代码,TODO 计划在 kernel 分歧修复后收紧阈值。

## 关联

#42430;失败构建:[Hybrid SSM NixlConnector PD accuracy tests (4 GPUs)](https://buildkite.com/vllm/ci/builds/66835/canvas?sid=019e3ed4-250f-4035-99b7-e6c0ba9162c1&tab=output)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
