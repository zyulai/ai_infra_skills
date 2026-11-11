# PR #48556: [Fix] moe: Fix illegal memory access in the W4A16/W8A16 Triton MoE kernel (unmasked K tail)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#48556](https://github.com/sgl-project/sglang/pull/48556) |
| 作者 | qhmiao |
| 合并日期 | 2026-11-10 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 1 · 💬 7 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
W4A16/W8A16 Triton MoE 内核权重加载缺 K 掩码,部分 K 尾块越界读显存导致 CUDA IMA;给权重加载补上 k_mask 修复。

## 问题现象
Weight-only 量化的 MoE 模型(W4A16 / W8A16)走 Triton MoE 路径时,在 `fused_moe_kernel_gptq_awq` 中崩溃,报 `CUDA error: an illegal memory access`。复现:AutoRound W4A16 量化的 Qwen3.8-Flash-Next 检查点,TP 4 下第一个 decode step 必现。作者称在 v0.5.22 上,每个这样的服务器都在首次 decode 时挂掉。

受影响内核的使用方:`--quantization moe_wna16`(GPTQ/AWQ 检查点);Marlin 无法承接的 AWQ 与 AutoRound MoE 层(如平台无 Marlin 支持,或 per-rank intermediate size 不是 max(64, group size) 的倍数);Compressed-tensors W4A16 MoE(`CompressedTensorsWNA16TritonMoE`:ROCm 上恒走,SM100/SM103 上对称 INT4 group 32/128 默认走,其他平台加 `--moe-runner-backend triton`)。

越界条件:per-rank K 不是所用 `BLOCK_SIZE_K` 的倍数。decode 批量(M ≤ E)默认 config 用 `BLOCK_SIZE_K` 64,per-rank intermediate size 不是 64 的倍数时 down projection 越界,常见如 GLM-4.5-Air/DeepSeek-V2-Lite/Qwen1.5-MoE-A2.7B(TP4 时 per-rank K=352)、Qwen3-30B-A3B(TP8 时 96)、Qwen3.8-Flash-Next(TP4 时 160)。是否真的崩溃取决于权重张量后面内存里是什么;作者实测 Qwen3.8-Flash-Next(512 experts)TP4 必现:down projection 权重每卡 `[512, 2560, 80]` uint8 恰好 100 MiB,越界读出了分配边界。

## 根因
正文披露:当 K 不是 `BLOCK_SIZE_K` 的倍数时,`fused_moe_kernel_gptq_awq` 会构建 `k_mask` 并应用于 activation、scale、zero-point 的加载,但漏掉了权重加载 `b = tl.load(b_ptrs)`。于是最后一个 K tile 会读过每行权重的末尾;在最后一个 expert 的最后一行,读过整个权重张量的末尾。多读的值会被掩码为 0 的 scale 相乘,所以只要读不触发 fault 结果仍正确;`fused_moe_kernel` 对同一处加载早已加了掩码。

## 修复方案
给权重加载套用已有的 `k_mask` / `k_other`(与 scale、zero-point 加载一致);当 K 是 `BLOCK_SIZE_K` 的倍数时 `k_mask` 为 `None`,加载行为不变。改动在 `python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py` 的 `fused_moe_kernel_gptq_awq`:

```diff
-        b = tl.load(b_ptrs)
+        b = tl.load(b_ptrs, mask=k_mask, other=k_other)
```

并新增 `test/registered/kernels/ops/moe/test_fused_moe_gptq_awq_k_tail.py`:① `test_partial_k_tile` 用 int4/int8、有无 zero-point、K=96/160 + `BLOCK_SIZE_K` 64 对比反量化后 matmul(现有 wna16 测试只覆盖 K=128/1024,部分 K 尾块此前无覆盖);② `test_weight_tail_at_allocation_end` 复刻 Flash-Next 的 down projection 形状(E 512, N 2560, K 160, group 32),权重最后分配使越界离开分配区,修复前在 L40S 上崩溃。

验证:1x L40S(SM89)、torch 2.14.1、Triton 3.8.0。修复前 `test_weight_tail_at_allocation_end` 4/4 次报 `CUDA error: an illegal memory access`;修复后 2/2 通过;compute-sanitizer memcheck 下修复前报 6 个 `Invalid __global__ read of size 1 bytes`(指向该权重加载),修复后 0 错误。

## 影响与回归风险
- 影响窗口:正文只写明在 v0.5.22 上复现,未给出明确的版本/提交范围。
- 回归面:K 为 `BLOCK_SIZE_K` 倍数(`even_Ks`)时生成代码不变;否则权重加载多出一个谓词,与同一循环中 scale、zero-point 加载已带的谓词相同。被改内核覆盖 moe_wna16、compressed-tensors W4A16 等 Triton MoE 路径;性能未做 benchmark。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
