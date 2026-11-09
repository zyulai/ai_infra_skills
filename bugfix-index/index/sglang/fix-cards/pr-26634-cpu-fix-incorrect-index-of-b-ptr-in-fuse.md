# PR #26634: [CPU] fix incorrect index of b_ptr in fused_sigmoid_gating_delta_rule…_update_kernel_impl

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26634](https://github.com/sgl-project/sglang/pull/26634) |
| 作者 | blzheng |
| 合并日期 | 2026-05-29 |
| labels | sgl-kernel, intel, cpu, run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

CPU fused_sigmoid_gating_delta_rule 更新核函数中 `b_ptr` 只按 head 索引、漏掉 batch 维,批内各请求 beta 值取错,改为按 batch 索引。

## 问题现象

`fused_sigmoid_gating_delta_rule_update_kernel_impl`(CPU mamba/fla 路径)在 batch_size > 1 时计算结果不正确。正文本身未描述具体现象,仅说明 bug 由 [PR #19484](https://github.com/sgl-project/sglang/pull/19484) 的 review 评论(@fadara01)指出,本 PR 予以修复。

## 根因

正文未披露根因细节,仅指认 `fused_sigmoid_gating_delta_rule_update_kernel_impl` 存在 bug(指认来自正文引用的 review 评论)。从 diff 看(由 diff 推断):核函数里 `beta_val = 1 / (1 + std::exp(-b_ptr[ni]))` 只用 head 索引 `ni` 取 `b_ptr`,完全没带 batch 偏移 `bi`,导致 batch 内所有请求都读到第 0 个 batch 元素的 beta 值;正确索引应为 `b_ptr[bi * v_num_heads + ni]`(与紧邻的 `v_offset`/`o_offset` 一致地包含 `bi` 分量)。

## 修复方案

1. `sgl-kernel/csrc/cpu/mamba/fla.cpp`:`fused_sigmoid_gating_delta_rule_update_kernel_impl` 中 beta 的取值索引从 `b_ptr[ni]` 改为 `b_ptr[bi * v_num_heads + ni]`;
2. `test/registered/cpu/test_mamba.py`:原测试固定 `batch_size = 1`(恰好掩盖此 bug),改为 `@parametrize(batch_size=[1, 4], ...)` 覆盖多 batch;配套修正测试里 `g`/`beta` 的 `unsqueeze` 维度(0→1)与 `query_start_loc` 构造(`[0, 1]` → `torch.arange(batch_size + 1)`)。

关键 diff 摘录:

```diff
-      float beta_val = 1 / (1 + std::exp(-b_ptr[ni]));
+      float beta_val = 1 / (1 + std::exp(-b_ptr[bi * v_num_heads + ni]));
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 CPU 后端的该单算子核函数;`batch_size == 1` 时 `bi = 0`,新旧索引等价,单请求行为不变;batch > 1 时输出值会发生变化(即修复本身)。测试改动仅涉及 CPU 注册测试。

## 关联

- [PR #19484 review 评论](https://github.com/sgl-project/sglang/pull/19484#discussion_r3295334788)(指出 bug 的来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
