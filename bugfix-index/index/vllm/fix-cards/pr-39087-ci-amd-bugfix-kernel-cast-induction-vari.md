# PR #39087: [CI][AMD][BugFix][Kernel] Cast induction variable to int64 on MI350 for chunk_gated_delta_rule_fwd_kernel_h_blockdim64 to avoid illegal memory access

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39087](https://github.com/vllm-project/vllm/pull/39087) |
| 作者 | rasmith |
| 合并日期 | 2026-04-08 |
| labels | bug, ready, rocm |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
Triton kernel 在 MI350 上因归纳变量 `i_t` 32 位地址计算越界报 illegal memory access;将其转为 int64 修复。

## 问题现象
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 这个 Triton kernel 在 MI350 上生成非法内存访问(illegal memory access)。正文未给出更多触发条件细节;验证方式为 `pytest -s -v models/language/generation_ppl_test/test_qwen.py`,修复后 `3 passed, 17 warnings in 82.43s`。

## 根因
正文披露:kernel 在 MI350 上出现非法内存访问,把归纳变量 `i_t` 转成 `int64` 后问题消失。正文未进一步解释底层机制。(由 diff 推断)`i_t` 此前以默认 32 位整型参与 `h + i_t * stride_h`、`last_idx = min((i_t + 1) * BT, T) - 1`、`i_t * BT + tl.arange(0, BT)` 等地址/索引计算,在 MI350 上(长序列、大 stride 场景)乘积超出 32 位范围导致越界寻址;显式 `.to(tl.int64)` 后地址计算以 64 位进行,不再溢出。

## 修复方案
改 `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` 的 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`:将 main recurrence 循环中 4 处 `make_block_ptr` 的基地址 `h + i_t * stride_h` 改为 `h + i_t.to(tl.int64) * stride_h`;并将 `last_idx` 计算的 `(i_t + 1) * BT` 与 USE_G 分支的 `i_t * BT + tl.arange(0, BT)` 中 `i_t` 同样转为 int64。

```diff
-            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
+            h + i_t.to(tl.int64) * stride_h,
...
-        last_idx = min((i_t + 1) * BT, T) - 1
+        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `chunk_delta_h.py` 中该 blockdim64 kernel 的索引计算;所有走 gated delta rule(如 Qwen 类 GDN 模型)在 MI350 上的前向状态递推路径;其他 blockdim 的 kernel 与非 ROCm 平台代码未动;int64 索引会改变寄存器/寻址开销,但数值语义不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
