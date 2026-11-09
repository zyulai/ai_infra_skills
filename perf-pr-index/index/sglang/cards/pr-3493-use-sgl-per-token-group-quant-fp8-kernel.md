# PR #3493: use sgl_per_token_group_quant_fp8 kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2025-02-12 |
| 分类 | Quantization | tier: major |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 3 个文件:`pyproject.toml`、`fused_moe.py`、`fp8_kernel.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/3493 |

## 一句话结论

fused MoE 的 per-token-group FP8 激活量化从 Triton 换成 sgl-kernel CUDA 实现,DeepSeek R1/V3 TP8 端到端吞吐与延迟均约提速 1%。

## 动机

DeepSeek-V3/R1 这类 FP8 block-quant 模型中,fused MoE kernel 每次调用前都要对激活 A 做 per-token-group FP8 量化,此前走 Triton 实现 `per_token_group_quant_fp8`。sgl-kernel 0.0.3.post4 提供了对应的 CUDA 实现 `sgl_per_token_group_quant_fp8`。该量化位于 MoE GEMM 的关键路径上,每个 MoE 层、每次 forward 都要执行,换成 CUDA kernel 可以削减这段固定开销——PR 正文未写更长的动机陈述,直接给出端到端对比。

## 优化点

改动很小且边界清晰。`fp8_kernel.py` 新增 wrapper `sglang_per_token_group_quant_fp8(x, group_size, eps=1e-10, dtype=fp8_type_)`:校验最后一维按 group_size 整除、x 连续,预分配 `x_q`(fp8)与 `x_s`(fp32,形状 `shape[:-1] + (shape[-1]//group_size,)`),然后一次调用 sgl-kernel 的 `sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max)` 完成量化;fp8_min/fp8_max 由 dtype 的 finfo 推出(e4m3fn)。`fused_moe.py` 的 `invoke_fused_moe_kernel` 在 block-quant 分支(`len(block_shape) == 2`)按平台分发:CUDA 用新 CUDA kernel,否则保持原 Triton `per_token_group_quant_fp8`,调用方语义(输入/输出形状断言)不变。`pyproject.toml` 把 sgl-kernel 下限提到 `>=0.0.3.post4` 以引入该符号。ROCm 路径与其他 `per_token_group_quant_fp8` 调用点在本 PR 中不动。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_kernel.py: 新增 CUDA wrapper sglang_per_token_group_quant_fp8
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py: invoke_fused_moe_kernel 的 block-quant 分支按 _is_cuda 分发
python/pyproject.toml: sgl-kernel >= 0.0.3.post4
```

```python
if _is_cuda:
    A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
else:
    A, A_scale = per_token_group_quant_fp8(A, block_k)
```

## 性能声明

条件:`launch_server --model deepseek-ai/DeepSeek-V3 --tp 8`(硬件 PR 未注明),`bench_serving --backend sglang --num-prompts 1000 --request-rate 8`,各跑两次;结论原文:"Both latency and throughput achieved an end-to-end speedup of approximately 1% in DeepSeek R1"。

- 第一次:Total token throughput 2088.07 → 2093.14 tok/s;Mean E2E 61,571.47 → 61,322.63 ms;Median E2E 51,259.16 → 51,756.45 ms(此格中位数略差);Total throughput(输入+输出)小幅提升,输出吞吐 802.61 → 804.56 tok/s。
- 第二次:Total token throughput 2120.10 → 2130.97 tok/s;Mean E2E 56,819.06 → 55,881.33 ms;Median E2E 47,473.95 → 46,573.34 ms;Mean TTFT 834.04 → 772.20 ms;Mean TPOT 465.60 → 447.66 ms;P99 ITL 2379.64 → 2161.58 ms。
- 输入/生成 token 数两次完全一致(301,701 / 188,375),可行;kernel 级量化微基准:PR 未提供。

## 验证与回归风险

测试:PR 未列单测或精度评估;量化语义(eps=1e-10、fp8_min=-fp8_max)与原 Triton wrapper 一致,输出张量形状相同,上游断言(`cdiv(A.shape[-1], block_k) == A_scale.shape[-1]` 等)保留。风险:仅 CUDA 生效,ROCm 保持 Triton 路径;要求 `x` 连续(assert);本 PR 只替换 fused MoE 的激活量化调用点,`per_token_group_group_quant_fp8` 的其他调用点不变;依赖 sgl-kernel 版本下限提升;输出 token retokenized 计数两侧略有差异(187,534/187,556 等),作者未讨论。

## 关联

无(正文未列其他 PR;为 sgl-kernel 量化 kernel 系列在 fused MoE 路径的首次接线之一)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
