# PR #34031: [CI][torch.compile] Fix incorrect filtering for E2E fusion tests on B200

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34031](https://github.com/vllm-project/vllm/pull/34031) |
| 作者 | ProExpertProg |
| 合并日期 | 2026-02-09 |
| labels | ready, ci/build |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
E2E fusion 改造(#33293)的过滤逻辑误删了应保留的 B200 测试;把多条 -k 过滤合并为含 FLASHINFER、inductor_partition、qwen3/llama-3 的正确表达式。

## 问题现象
B200 上 torch.compile 的 E2E fusion CI 测试覆盖不完整:#33293 最终版把一些本应保留的测试切掉了(slashing some tests that should have remained),导致部分模型/后端组合不再被 CI 执行。正文未给出具体被删测试清单。

## 根因
正文披露:E2E fusion overhaul(#33293)的最终版本在精简时错误地删除了应保留的测试,过滤逻辑不正确;本 PR 通过改进过滤逻辑修复,且仅针对 B200 测试。

## 修复方案
改 `.buildkite/test_areas/compile.yaml` 中 B200 相关步骤的 pytest `-k` 过滤表达式,把原先多次传 `-k`/分条注释的组合方式改写为单一布尔表达式,并补上 FLASHINFER 注意力后端与 qwen3+quant_fp8 组合:
- Fusion E2E TP1 Quick(B200):`test_tp1_quant.py` 由三条 `-k`(inductor_partition 非 rms_norm 非 quant_fp8;inductor_partition+quant_fp8+qwen3;llama-3)合并为 `-k "inductor_partition and (FLASHINFER and not +rms_norm and (not +quant_fp8 or +quant_fp8 and qwen3) or llama-3)"`。
- Fusion E2E TP2 Config Sweep(B200):`test_tp2_ar_rms.py` 与 `test_tp2_async_tp.py` 同样改为含 FLASHINFER 的合并表达式(ar_rms 另保留 `or Llama-3.1-8B-Instruct-FP4` sweep)。
- 顺带修正注释:TP2 Quick 的 llama3 说明由「fp4 & fp8 & bf16」改为「fp8 & bf16」。

```diff
-    - pytest -v -s tests/compile/fusions_e2e/test_tp1_quant.py -k "inductor_partition and not +rms_norm and not +quant_fp8" -k "inductor_partition and not +rms_norm and +quant_fp8 and qwen3" -k "llama-3"
+    - pytest -v -s tests/compile/fusions_e2e/test_tp1_quant.py -k "inductor_partition and (FLASHINFER and not +rms_norm and (not +quant_fp8 or +quant_fp8 and qwen3) or llama-3)"
```

测试计划为 CI(Test Result 标注 TBD)。

## 影响与回归风险
- 影响窗口:正文未说明(与 #33293 的 E2E fusion 改造相关)。
- 回归面:仅 Buildkite CI 配置中 B200 的 fusion E2E 测试选择逻辑;表达式写错可能导致测试过多或过少(以 CI 实际运行为准),不影响生产代码。

## 关联
- [PR #33293](https://github.com/vllm-project/vllm/pull/33293)(E2E fusion overhaul,过滤错误的来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
