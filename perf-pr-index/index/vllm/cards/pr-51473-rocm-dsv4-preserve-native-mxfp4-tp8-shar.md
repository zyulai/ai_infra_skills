# PR #51473: [ROCm][DSV4] Preserve native MXFP4 TP8 shard allocation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Fangzhou-Ai |
| 合并 | 2026-08-11 |
| 分类 | Quantization（tier: major） |
| Label | rocm, quantization |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/51473 |

## 一句话结论

去掉 DSV4 MoE TP8 MXFP4 shard 的 I512 物理填充，每 rank 省 31.25 GiB 模型显存、KV token 容量 +82.76%。

## 动机

DeepSeek-V4 的 routed-expert intermediate 宽度为 3072，TP8 下每卡原生 shard 是 384 宽（I384）。通用 ROCm 分配路径在参数分配与 checkpoint 加载前把它向上取整到 512（I512）——这是物理填充而非 sharding 失败，导致 routed-expert 显存分配膨胀 33%，挤占 KV cache 容量。

## 优化点

把取整规则从各处特例收拢进通用 MXFP4 oracle：`Mxfp4MoEMethod` 将 activation 传入 rounding oracle，新增一条规则——AITER BF16/SiLU 后端无论模型与 routing method 如何，一律使用 128 对齐布局，使 I384 这类原生 shard 在 TP8 下保持原生宽度；其他 activation/backend 保持既有对齐。由此不需要任何 AMD 特有的 `RoutedExperts` 子类、量化方法或模型接线；EP checkpoint 加载与 expert mapping 不变；W2 E8M0 scale 列保留其独立的 12-to-16 安全填充。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py: 新增 AITER_MXFP4_BF16+SiLU 的 128 对齐取整规则
tests/kernels/moe/test_ocp_mx_moe.py: 回归测试 test_aiter_mxfp4_bf16_silu_preserves_native_tp_shard
```

```python
elif (
    backend == Mxfp4MoeBackend.AITER_MXFP4_BF16 and activation == MoEActivation.SILU
):
    # AITER's A16W4 SiLU kernel handles native dimensions aligned to 128.
    intermediate_size = round_up(intermediate_size, 128)
    hidden_size = round_up(hidden_size, 128)
```

## 性能声明

显存（8x gfx950，TP1 x DP8，EP-off，对照 parent `821717118f`）：Model/rank 156.45→125.20 GiB（省 31.25 GiB），KV/rank 39.31→71.85 GiB，KV tokens/rank 2,859,796→5,226,632（+82.76%）；I384 分配与 checkpoint header 预测差 0.61% 内。EP8 另测：124.61 GiB 模型、69.79 GiB KV、5,076,613 KV tokens/rank（有意不变）。

8K/1K（ISL 8192 / OSL 1024，C64 配对跑，prefix caching 关闭）：Output throughput 574.60→589.75 tok/s（+2.64%），Median TPOT -2.72%、P90 TPOT -3.37%，通过预声明的 ≥95% 吞吐、≤10% TPOT/E2E 回归门槛。

TP8 concurrency sweep（同 workload）：C=1/4/16 微增（+0.36%/+0.92%/+0.69%），C=64 +5.56%（10,055.37→10,614.10 tok/s），mean TPOT -4.99%、E2E -5.30%，C=64 mean TTFT -13.17%。

Agentic C64（TP1 x DP8，1 小时，prefix caching 开启，1M 上下文）：Successful requests 1,370→2,020（+47.45%），Total throughput 47,808.59→64,163.00 tok/s（+34.21%），Output throughput +51.92%，Mean TTFT -39.51%，P99 TPOT -73.55%；实测 prefix hit 93.82%→95.79%。

模型评估：GSM8K 1319 样本 5-shot，strict exact match 0.962092。PR 自注：agentic 配对测试未隔离"额外 KV 容量"与"物理 MoE 宽度减小"两者的贡献（都是去掉 I512 填充的直接效果）。

## 验证与回归风险

- 新增 ROCm/AITER 条件回归测试钉住 I384 原生分配（TP8、384 experts、intermediate 3072 → rounded 384）；EP 权重过滤测试 `test_384_experts_ep8` 通过；pre-commit（Ruff/mypy/SPDX 等）全过。
- 回归风险：已存在的 vLLM `sharded_state` 快照若按旧 256 对齐（I512）存储 AITER BF16/SiLU 张量，物理形状变化后必须重新生成（DSV4 TP8 即 I512→I384）；vLLM 仍按 `gpu_memory_utilization` 预留总显存，节省显存转化为 KV 容量而非降低最终 rocm-smi 分配。

## 关联

- 相邻对齐规则前序：PR #50597（Kimi-K3 AITER/SiTU 128 对齐规则，本 PR 将同一 backend 对齐策略扩展到 SiLU 且不带模型特例）；PR 正文还对照了 #46201（gpt-oss 运行时 pad）、#50928（Kimi-K3 模型级 shard sizing）、#48728（未合并 FHMoE）、#51217（masked activation）以说明无功能重复。
- 依赖项：agentic 测试两臂均含已合并的 packed-KV zeroing 修复 PR #50276（已在 upstream main，不属于本 PR）。
- 同分类：本批 PR #61457（Quantization，per-layer fused shared expert）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
