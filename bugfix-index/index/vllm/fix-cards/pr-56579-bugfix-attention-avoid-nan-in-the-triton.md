# PR #56579: [Bugfix][Attention] Avoid NaN in the Triton softcap for large attention logits

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56579](https://github.com/vllm-project/vllm/pull/56579) |
| 作者 | kushaldabbe |
| 合并日期 | 2026-09-22 |
| labels | bug, ready, verified |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

Triton softcap 用两个 exp 相除实现 tanh,|S/x|>~88 时 exp 溢出为 inf/inf=NaN 污染整行注意力;改为单 exp 形式 `1 - 2/(exp(2y)+1)` 干净饱和修复。

## 问题现象

注意力 logits 较大时 softcap 输出 NaN。作者在 GTX 1650 Ti 上用真实 helper 验证:输入 `[1.0, 100.0, 1000.0, 3000.0, 5000.0, 10000.0, -1.0, -10000.0, 4400.0]`,修复前输出 `[0.9999, 48.2, 50.0, 50.0, NaN, NaN, -0.9999, NaN, inf]`,即分数超过约 88 * soft_cap 就溢出为 inf/NaN,整个注意力行被污染。Gemma-2 风格模型设置 `attn_logit_softcapping = 50`,pre-softcap 分数超过 4400 即触发——这类模型注意力 logits 本来就非常大,这正是它们需要 softcap 的原因。

## 根因

正文披露:`apply_softcap` 把 `x * tanh(S / x)` 实现为 `x * (exp(S/x) - exp(-S/x)) / (exp(S/x) + exp(-S/x))`。当 `|S / x| > ~88` 时两个指数都溢出到 `inf`,比值变成 `inf / inf = NaN`,污染整行注意力输出。

## 修复方案

修改 `vllm/v1/attention/ops/triton_attention_helpers.py` 的 `apply_softcap`:改用恒等式 `tanh(y) = 1 - 2 / (exp(2y) + 1)`,单个指数对正负号都能在 ±1 干净饱和,不会溢出。关键 diff:

```diff
-    Sdiv = S / x
-    p1 = tl.exp(Sdiv)
-    p2 = tl.exp(-Sdiv)
-    return x * (p1 - p2) / (p1 + p2)
+    y = S / x
+    return x * (1.0 - 2.0 / (tl.exp(2.0 * y) + 1.0))
```

该 helper 被 unified、DiffKV、INT4 三个注意力 kernel 共享,一处修复全部受益。同时在 `tests/kernels/attention/test_triton_unified_attention.py` 新增 `test_softcap_does_not_overflow_on_large_scores`,断言大分数下输出全部有限且与 `soft_cap * torch.tanh(...)` 参考一致;修复后输出 `[0.9999, 48.2, 50.0, 50.0, 50.0, 50.0, -0.9999, -50.0, 50.0]`。

## 影响与回归风险

- 影响窗口:正文未说明具体版本;受影响的是使用 Triton softcap 且 `attn_logit_softcapping` 非零(如 Gemma-2 系)的模型。
- 回归面:`apply_softcap` 是共享 helper,改动会进入 unified/DiffKV/INT4 三个 Triton kernel 的数值路径;新公式与 tanh 数学等价,仅在原实现溢出的区域行为不同。测试结果显示 336 个 softcap 测试通过;fp8 用例仅在 SM75 上失败(Triton 报告不支持 `fp8e4nv`),作者认为与本次改动无关的平台限制。新测试在未修复的 main 上确认失败、修复后通过。

## 关联

- Fixes #56578

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
