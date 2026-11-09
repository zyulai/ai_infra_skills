# PR #39705: [Bugfix][Kernel][ROCm] Fix triton_w4a16 scales mismatch when BLOCK_K > group_size

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39705](https://github.com/vllm-project/vllm/pull/39705) |
| 作者 | JartX |
| 合并日期 | 2026-04-13 |
| labels | bug, ready, rocm |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
Triton W4A16 GEMM 在 BLOCK_K > group_size 时一个 tile 跨多个 scale 组却只套第一组 scales,静默损坏尾部行反量化权重;钳制 BLOCK_K。

## 问题现象
实际使用中表现为上下文超过 10K token 后 agent 工具调用出错:模型常以相同参数循环调用工具、重复执行同一任务、任务做一半丢下,或(编码 agent 场景)开始删除相关代码、幻觉出指令。作者此前在 ROCm RDNA3 上把这类错误当成模型本身的问题(当时被迫用 float16 + Group Size 的 ExLlamaLinearKernel)。改用 #37352 引入的 kernel 以 bfloat16 运行 Qwen3.5-35B-A3B-GPTQ-W4A16-G32 后错误反而更频繁,才定位到本 bug;修复后该模型"几乎不再犯错、精度大幅提升"。

## 根因
正文披露:`triton_w4a16_gemm_kernel` 每个 BLOCK_K tile 只按单一 group index 加载量化 scales 和 zero points,而 kernel launcher 不强制 `BLOCK_K <= group_size`;当 BLOCK_K 超过 group_size 时,一个 tile 跨越多个 scale 组,但只有第一组的 scales 被应用到 tile 内所有行,tail 行的反量化权重被静默损坏。

## 修复方案
`vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py` 的 `triton_w4a16_gemm` 在启动 kernel 前钳制:

```diff
+    # The kernel loads scales/zeros for a single group per BLOCK_K tile
+    # (one g_idx per iteration). If BLOCK_K > group_size, rows at the tail
+    # of the tile dequantize with the wrong group's scales, silently
+    # corrupting the output. Clamp BLOCK_K to group_size to keep one
+    # scale group per tile.
+    if group_size < BLOCK_K:
+        BLOCK_K = group_size
```

保证每个 tile 至多覆盖一个 scale 组。

## 影响与回归风险
- 影响窗口:#37352 引入该 Triton W4A16 kernel 起至本 PR(此前 ExLlama 路径是否受影响正文未说明)。
- 回归面:仅 `triton_w4a16_gemm` 的 BLOCK_K 选择——group_size < BLOCK_K 的配置(如 G32 搭默认 128/64 tile)BLOCK_K 变小,可能改变 GEMM 分块性能;group_size >= BLOCK_K 的配置与数值结果(本就正确的情况)不变。

## 关联
- #37352(引入该 kernel 的 PR,正文链接给出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
