# PR #30079: [MoE] Fix moe_fused_gate out-of-range expert id on all-NaN rows (fixes eagle_dp_attention crash)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30079](https://github.com/sgl-project/sglang/pull/30079) |
| 作者 | ch-wan |
| 合并日期 | 2026-07-04 |
| labels | jit-kernel |
| 热度 | 👍 2 · 💬 9 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论

`moe_fused_gate` 在全 NaN gating 行上选出越界专家 id(N+1 哨兵),使 `moe_align_block_size` 越界访存、EAGLE3+DP-attention 抓图崩溃;top-k 排序前将 NaN 映射为有限下界修复。

## 问题现象

`test/registered/spec/eagle/test_eagle_dp_attention.py`(`TestEAGLE3EngineDPAttention`)在 scheduled full run 中,decode CUDA-graph capture 阶段把 server 崩掉,崩溃点为 `count_and_sort_expert_tokens_kernel`(即 `moe_align_block_size` 的 sort),错误为 `CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS`。触发场景是 EAGLE3 + DP-attention graph capture 中为 garbage padding 位置产生的 all-NaN gating 行;非 NaN 的正常路由不受影响。

## 根因

正文披露完整链路:PR #29771(issue #26771)将 unified Triton router 设为 ungrouped softmax/sigmoid MoE 的默认(`SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=True`),qwen3-moe 的 topk 因此从 AOT `topk_softmax` kernel 切换到 `moe_fused_gate`。在全 NaN gating 行上,`max_val` 为 NaN,每一 lane 上 `cur == max_val` 均为 false,top-k 平局裁决 `tl.min(lane_id)` 一路落空,最终输出 `N + 1` 哨兵值——对 128 个专家即越界的 expert id 129。该越界 id 使 `moe_align` 索引 `cumsum_buffer[topk_id + 1]` = 130,而 `cumsum_buffer` 大小为 `num_experts + 2`(130),于是越界访存,触发 illegal address。旧的 AOT kernel 能容忍 NaN 行,所以这是潜伏缺陷,直到默认路由被切换才暴露。

## 修复方案

在 top-k 排名之前把 NaN 映射为一个有限下界(正文表述为 map NaN -> -inf;实际实现取 `-1e30`),使退化行也能选出在范围内的专家 id,与 AOT kernel 行为一致。有限值与 `-inf` 不受影响,正常行结果 bit-identical。

关键摘录(`python/sglang/jit_kernel/moe_fused_gate.py`,全部改动仅此 3 行):

```diff
+    # Map NaN -> a finite floor
+    biased = tl.where(biased == biased, biased, -1e30)  # [BLOCK_M, BLOCK_N]
```

正文给出的 H200 standalone 验证:all-NaN / Inf gating 行输出回到 `[0, num_experts-1]`(此前输出 129);`moe_fused_gate -> moe_align_block_size` 全管线在 all-NaN 输入下不再 fault;跨 token 数/seeds/dtypes 的正确性 sweep 与 torch 参考一致、无变化;router→align 管线的 CUDA-graph capture + replay 干净。

## 影响与回归风险

- 影响窗口:自 #29771 将 unified Triton router 设为默认(ungrouped softmax/sigmoid MoE,如 qwen3-moe)起,至本 PR 合并前。
- 回归面:按 diff,仅 `moe_fused_gate` Triton kernel 中一行 NaN 映射,作用于退化(padding)行;正常路由路径的数值行为保持不变(bit-identical),风险极低。

## 关联

- PR [#29771](https://github.com/sgl-project/sglang/pull/29771)(引入默认切换)
- issue [#26771](https://github.com/sgl-project/sglang/issues/26771)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
