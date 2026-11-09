# PR #62843: [Perf][GLM-5.3] Enable GLM low-latency BF16 GEMM on SM107

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @csahithi |
| 合并 | 2026-10-22 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, deepseek, glm |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/62843 |

## 一句话结论

在 SM107（VRNVL72）启用 GLM BF16 低延迟 GEMM，kernel 快 1.2×–1.9×，端到端 c1 吞吐 +8.1%。

## 动机

GLM-5.2/5.3 的低延迟 BF16 GEMM 选择（CuTe DSL skinny GEMM + `dsv3_fused_a_gemm`，用于 decode batch sizes）此前只在 SM103 启用，导致 VRNVL72（SM107）上每个未量化 decode projection 都走 cuBLAS。本 PR 用在 VRNVL72 上 sweep 出的表把该选择开放到 SM107，方式与 #54606 为 Kimi-K3 做的相同。

## 优化点

按 local `(N, K)` 权重形状与 token 数 `M` 分派：新增 `GLM52_PROJECTIONS_SM107` 表，只在 CuTe 快于 cuBLAS 超过 0.15 µs 噪声带的 `(shape, M)` 处登记 CuTe DSL skinny GEMM / `dsv3_fused_a_gemm` 配置，其余走 cuBLAS。覆盖 TP4 的 fused qkv_a、q_b / indexer wq_b、o_proj（M ≤ 3）、indexer wk + weights_proj、shared-expert gate_up / down 与 MTP eh_proj；TP8 q_b 沿用 SM103 行并在 SM107 复测，其 CuTe 覆盖 M=1–2、`dsv3_fused_a_gemm` 覆盖 M=3–16。表中每个 `(shape, M)` 仅在 CuTe 快于 cuBLAS 超过 0.15 µs 噪声带时登记，其余 M 仍走 cuBLAS。SM103 保持原表与路由，其他设备仍用默认 GEMM。

## 关键改动

```text
vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py: 新增 SM107 表与各投影 CuTe/dsv3 配置，docstring 说明按 (N,K)/M 分派
tests/kernels/test_bf16_skinny_gemm.py: 形状测试覆盖 SM103/SM107 两表并集与 capability (10,7) 路由
```

```python
-"""GLM-5.2 decode GEMM selection for unquantized BF16 on SM103."""
+"""GLM-5.2 decode GEMM selection for unquantized BF16 on SM103/SM107.
+
+Dispatch is by local ``(N, K)`` weight shape and token count ``M``.
+:data:`GLM52_PROJECTIONS` was measured on B300 (SM103) and
+:data:`GLM52_PROJECTIONS_SM107` on VRNVL72 (SM107). The SM107 table also covers
+the TP4 projections (q_b, o_proj, shared expert) and the replicated indexer
+projections, which run as BF16 GEMMs at decode batch sizes.
+"""
```

## 性能声明

Kernel 微基准（单 VRNVL72 GPU、BF16、CuTe DSL skinny GEMM vs cuBLAS，仅列 CuTe 胜出项）：fused qkv_a 5.84/6.91 µs（1.18×）、q_b 4.31/5.71（1.32×）、o_proj 5.90/7.77（1.32×）、indexer wk 2.83/5.43（1.92×）、shared gate_up 3.96/5.40（1.36×）、shared down 3.19/4.28（1.34×）、MTP eh_proj 12.90/16.96（1.31×）；正文概述每个形状在目标 M 上比 cuBLAS 快 **1.2× to 1.9×**。端到端 A/B（GLM-5.3-NVFP4、TP4、单 VRNVL72 节点、8k1k recipe、aiperf synthetic 8000 in/1000 out、ignore_eos、stock 与 PR 交替）：无 spec decode 时输出吞吐 c1 **+8.1%**（ITL 6.35→5.86 ms）、c2 +5.9%、c4–c8 +2.2%；MTP K=3 时 c1 +8.5%（单次运行）、c2 +2.0%、c4 +1.3%。GSM8K 0.9735 stock vs 0.9773（run-to-run noise 内）。

## 验证与回归风险

`pytest tests/kernels/test_bf16_skinny_gemm.py -k glm`；GPU 形状测试覆盖两表并集与 SM107 `dsv3_fused_a` 范围；`ruff check`/`ruff format --check`/`typos` 通过。nsys decode step profile（GLM-5.3-NVFP4 TP4、MTP K=3、c=1）：step time 6.97→6.87 ms、kernels/step 2004→1801、dense-GEMM kernel time 4355→3859 µs；c=1 时 routed-MoE 辅助流是关键路径，GEMM 收益未完全转化为 step time。风险面：MTP 行每 arm 仅一次运行、temperature 1 下 acceptance 有波动（mean AL 1.58 stock vs 1.61 PR）；B300 上 tests/routing/GSM8K/throughput 不变；SM103 保持原表。

## 关联

PR 正文提及 #54606（Kimi-K3 低延迟 GEMM 在 SM107 的先例）。素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
