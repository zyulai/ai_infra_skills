# PR #21348: Fix MxInt4 MoE returning wrong output variable

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21348](https://github.com/sgl-project/sglang/pull/21348) |
| 作者 | sshleifer |
| 合并日期 | 2026-03-26 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
MxInt4 MoE 把结果写入 symm_output 却返回未填充的 output 变量;改为直接返回 symm_output 修正错误输出。

## 问题现象
正文未描述具体复现步骤与影响面,仅指出:`trtllm_mxint4_block_scale_moe` 把结果写入通过 `output=` kwarg 传入的 `symm_output` buffer,而其返回值是一个 `CombinedOutput` 所不期望的 list。旧代码把这个返回值当作 MoE 输出继续传递,走 compressed-tensors w4a4 MxInt4 量化方案的 MoE 前向即拿到错误结果。

## 根因
正文披露:`apply_weights` 中把内核返回值赋给局部变量 `output`,并以 `StandardCombineInput(hidden_states=output)` 返回;但真实计算结果写在 `output=` kwarg 传入的 `symm_output` 里,返回的 list 类型也不是 `CombinedOutput` 期望的,变量用错即 bug 所在。

## 修复方案
改 `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py` 的 `apply_weights`:不再接收内核返回值,改用真正被写入的 `symm_output` 构造返回。

```python
-        output = trtllm_mxint4_block_scale_moe(
+        trtllm_mxint4_block_scale_moe(
             ...
             output=symm_output,
         )
-        return StandardCombineInput(hidden_states=output)
+        return StandardCombineInput(hidden_states=symm_output)
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:仅影响 compressed-tensors w4a4 MxInt4 MoE 的 `apply_weights` 返回路径;走该方案的模型输出从错误变量改为真实的 `symm_output`,其余量化/MoE 路径不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
