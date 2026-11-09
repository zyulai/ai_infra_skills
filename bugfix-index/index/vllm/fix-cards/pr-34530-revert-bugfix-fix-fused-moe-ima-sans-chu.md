# PR #34530: Revert "[Bugfix] Fix fused MoE IMA (sans chunking) by using int64 for strides"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34530](https://github.com/vllm-project/vllm/pull/34530) |
| 作者 | mgoin |
| 合并日期 | 2026-02-13 |
| labels | bug |
| 热度 | 👍 1 · 💬 0 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
回滚 #34279:int64 stride 修复引发大幅性能回退,恢复原 fused MoE kernel 签名待重新分析。

## 问题现象
本 PR 是回滚,正文未描述新的 bug 现象;触发条件是「#34279 合入后出现用户报告的大幅性能退化(large performance degradations reported)」。即 fused MoE kernel 因被 #34279 改为显式 `tl.int64` stride 后性能明显下降。正文未给出具体性能数据或日志。

## 根因
正文披露的根因:#34279 为修复 fused MoE 的 illegal memory access(IMA)而把 kernel stride 参数全部声明为 `tl.int64`,但这一改动带来了大幅性能退化,因此整体回滚,后续「with more careful performance analysis」再找类似效果的方案。更细的机制:`tl.int64` 显式标注会强制 Triton 以 64 位整数做地址计算,增加寄存器/指令开销,拖慢 kernel(由 diff 推断)。原 IMA 问题的根因正文未披露。

## 修复方案
回滚 PR #34279。`vllm/model_executor/layers/fused_moe/fused_moe.py` 中 `fused_moe_kernel_gptq_awq` 与 `fused_moe_kernel` 两个 Triton kernel 的 stride 参数去掉 `: tl.int64` 类型标注,恢复由 Triton 自动推断,例如:

```diff
-    stride_am: tl.int64,
-    stride_ak: tl.int64,
-    stride_be: tl.int64,
+    stride_am,
+    stride_ak,
+    stride_be,
```

`fused_moe_kernel` 中其余 stride(`stride_cm/stride_asm/stride_bbe` 等)同样去标注。

## 影响与回归风险
- 影响窗口:#34279 合入到本回滚之间的版本;期间所有走 fused MoE kernel 的 MoE 推理都承担了性能损失。
- 回归面:改动覆盖 GPTQ/AWQ 变体与通用 `fused_moe_kernel` 的全部 stride 参数,影响所有 MoE 模型的该 kernel 路径。回滚后 #34279 试图修复的 IMA 问题(非 chunking 场景)可能重新出现——正文称之后会「search for a similar result with more careful performance analysis」,即该 bug 的正式修复被推迟。

## 关联
- vllm-project/vllm#34279(被回滚的 PR,原 IMA 修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
