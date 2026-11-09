# PR #34507: [Bugfix] Fix fused MoE int32 overflow in stride*offset without perf regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34507](https://github.com/vllm-project/vllm/pull/34507) |
| 作者 | haosdent |
| 合并日期 | 2026-02-17 |
| labels | bug, ready |
| 热度 | 👍 3 · 💬 9 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
fused MoE Triton kernel 在大 M 时 stride*offset 溢出 int32 崩溃;仅把 offs_token 提升为 int64 修复,避免全 stride 扩位带来的 ~60x 性能回归。

## 问题现象
当禁用 chunking 且 M( token 数)很大时,fused MoE kernel 中 `stride_cm * offs_token` 乘积超过 int32 上限,导致 int32 溢出崩溃(issue #34413)。正文给出的触发量级:M=100000、n=2048、topk=6 时,`N * m * topk = 4096 * 600000 = 2.46B > int32 max`(约 2.147B)。此前的修复尝试 PR #34279 曾把所有 stride 参数标注为 `tl.int64`,但在小显卡(如 NVIDIA GB10)上因寄存器压力造成约 60x 性能回归,随后被 PR #34530 回退,溢出崩溃随之重新暴露。

## 根因
PR 正文披露:溢出源于 `stride_cm * offs_token`,其中 `stride_cm = N = w1.size(1)`,`offs_token` 最大可达 `M * topk`;禁用 chunking 且 M 大时二者乘积超出 int32 最大值。最早由 #34279 试图通过把全部 stride 加宽为 int64 修复,但因寄存器压力引发 ~60x 性能回归(小 GPU 如 NVIDIA GB10)被 #34530 回退,问题(#34413)重新出现。正文还附上全部 stride*offset 乘积的溢出安全审计表,确认其余表达式(如 `offs_k * stride_ak` 结果仅 ~128)在 int32 下安全,只有 `stride_cm * offs_token` 一处需要处理。

## 修复方案
不再加宽所有 stride,而是在加载 `offs_token` 后立即 cast 成 int64,利用 Triton 类型提升规则(int32 * int64 -> int64)使乘积自动变为 int64,沿用 `off_experts`、`offs_bn` 已有的 cast 模式,寄存器影响最小。改动 `vllm/model_executor/layers/fused_moe/fused_moe.py` 的两个 kernel:

```diff
-    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
+    # Cast to int64 to prevent overflow in stride*offset products
+    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
```

```diff
+    # Cast to int64 to prevent overflow in stride*offset products
+    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
+    offs_token = offs_token.to(tl.int64)
```

分别作用于 `fused_moe_kernel_gptq_awq`(加载时 cast)与 `fused_moe_kernel`(计算后 cast)。新增回归测试 `test_fused_moe_int64_overflow`:禁用 chunking(`VLLM_FUSED_MOE_CHUNK_SIZE=10000000`)、M=100000/n=2048/topk=6,并断言测试参数确实触发溢出条件(`N * m * topk > 2**31 - 1`),与 torch_moe 参考实现对比正确性;内存不足 12GB 时跳过。新测试在 main 上失败、本 PR 上通过。

## 影响与回归风险
- 影响窗口:PR #34530 回退 #34279 之后到本 PR 之前的 main 分支(溢出崩溃重新暴露);更早由 #34413 报告。
- 回归面:仅 fused MoE 两个 Triton kernel(`fused_moe_kernel`、`fused_moe_kernel_gptq_awq`)的 `offs_token` 类型变宽,影响这两个 kernel 的所有调用路径;相比 #34279 的全 stride 扩位,寄存器压力影响小得多。测试需要 ~12GB GPU 显存,低显存环境会跳过该回归测试。

## 关联
- https://github.com/vllm-project/vllm/issues/34413
- https://github.com/vllm-project/vllm/pull/34279
- https://github.com/vllm-project/vllm/pull/34530

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
