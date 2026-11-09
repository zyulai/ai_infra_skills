# PR #17461: Fix assertion tolerance for bf16 precision in triton attention UT

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17461](https://github.com/sgl-project/sglang/pull/17461) |
| 作者 | rahulvijayaraghavan |
| 合并日期 | 2026-03-03 |
| labels | intel, ready-for-review, run-ci |
| 热度 | 👍 0 · 💬 8 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
triton attention UT 的 `torch.allclose` 只设 rtol 没设 atol,近零值容差仅 1e-8 低于 bf16 精度,测试偶发失败;补 `atol=1e-3` 修复。

## 问题现象
`test_triton_attention_kernels.py::test_extend_attention` 偶发在第二轮迭代的 `torch.allclose()` 检查处抛 `AssertionError`,尽管 `o_extend` 与 `o_redundant` 的实际数值差在 bf16 可接受范围内。失败被 setup 中的固定 seed 掩盖;去掉固定 seed(`self._set_all_seeds(42)` 改为 pass)后:CUDA 连跑 10 次通过率 50%(5/10),XPU 连跑 10 次通过率 60%(6/10)。

## 根因
PR 正文披露:断言用的是 `torch.allclose(o_extend, o_redundant, rtol=1e-2)`,未设 `atol`,默认 `atol=1e-8`。`torch.allclose` 判据为 `|a - b| <= atol + rtol * |b|`,当张量值接近零时容差退化为约等于 `atol`(因 `rtol * |b|` ≈ 0),而 1e-8 远低于 bfloat16 的精度:bfloat16 epsilon 约 0.0078(7.8e-3),实测最大绝对差约 0.001(1e-3)。0.001 的差完全在 bf16 舍入误差预期内,但近零值处的容差只有 1e-8,断言因此偶发失败。

## 修复方案
在 `test/registered/attention/test_triton_attention_kernels.py` 的 `_test_extend_attention_once` 中,给两处 `torch.allclose()` 断言都加 `atol=1e-3`,在容纳 bf16 有限精度的同时仍能捕捉有意义的数值差异:

```diff
-        self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2))
-        self.assertTrue(torch.allclose(o_extend_mask, o_redundant, rtol=1e-2))
+        self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2, atol=1e-3))
+        self.assertTrue(
+            torch.allclose(o_extend_mask, o_redundant, rtol=1e-2, atol=1e-3)
+        )
```

调整容差后 CUDA 连跑 10 次通过率 100%(10/10)。

## 影响与回归风险
- 影响窗口:正文未说明;仅影响该单元测试文件。
- 回归面:改动只放宽测试断言的绝对容差,不触及任何产品代码或 kernel。风险是把 1e-3 到原判据之间的真实数值差异误判为通过;正文论证 1e-3 仍在 bf16 舍入误差量级内且远小于 epsilon(7.8e-3),rtol=1e-2 对大值判据不变。XPU(Intel)平台为该测试的主要受益方(labels 含 `intel`)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
