# PR #30655: Fix rel_diff being nan for bitwise-identical tensors

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30655](https://github.com/sgl-project/sglang/pull/30655) |
| 作者 | fzyzcjy |
| 合并日期 | 2026-07-09 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
逐位相同张量的 `rel_diff` 因 0/0 返回 NaN,令 `rel <= 0` 精确匹配判据反而失败;`max_abs_diff` 为 0 时直接定义为 0.0。

## 问题现象
对两个逐位相同(bitwise-identical)的张量,`calc_rel_diff` 计算出 0/0 并返回 NaN。任何与 NaN 的比较都是 False,因此在 predicate DSL 下,严格的 `rel <= 0` 判据(精确匹配检查,例如 deterministic-mode dumps)恰恰在完美匹配的张量上虚假失败——包括合法的全零张量,例如被饿死(starved)MoE experts 的梯度。

## 根因
正文披露:张量完全相同时相对差的分子与分母同时为 0,`calc_rel_diff` 算出 0/0 得 NaN;而 max_abs_diff 为 0 就意味着两张量相同,此时相对差应定义为 0.0 而不是去计算 0/0 比值。

## 修复方案
`python/sglang/srt/debug_utils/comparator/tensor_comparator/comparator.py` 的 `compute_diff`:先算出 `max_abs_diff`,为 0 时 `rel_diff` 直接取 `0.0`,否则才调用 `calc_rel_diff`。相同张量的 rel_diff 报告值相应由 NaN 变为 0.0。

```python
max_abs_diff = raw_abs_diff.max().item()
rel_diff = (
    0.0 if max_abs_diff == 0.0 else calc_rel_diff(x_target, x_baseline).item()
)
```

并新增回归测试 `test_bitwise_predicate`:张量与其 clone 比较时 `rel <= 0` 通过,sign-flipped 的 near-zero 张量对仍然失败。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:调试工具 tensor comparator 的 `rel_diff` 输出与 predicate 判定结果;仅 `max_abs_diff == 0` 的分支行为变化(NaN → 0.0),差异非零的路径不变;新增的回归测试覆盖 `rel <= 0` 的通过/拒绝两侧。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
