# PR #54560: [Kernel][Qwen] Add Hopper LL-GEMM tuning table for Qwen4Exp

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zigzagcai |
| 合并 | 2026-09-01 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/54560 |

## 一句话结论

为 Qwen4Exp decode 加 H200 SM90 LL-GEMM 调优表：E2E 输出吞吐 +14.15%、中位 TPOT -12.75%。

## 动机

Qwen4Exp / Qwen3.8-Flash-Next 的 decode 在 Hopper（SM90）上缺少独立调优的 LL-GEMM（low-latency skinny GEMM）表，此前要么复用 SM103 的表、要么把 Hopper 限制在 `M=1`。按 review 反馈，早前的 Hopper-only HyperConnection fusion 已移除，最终 diff 只保留调优表，用现有共享 CuTe DSL skinny-GEMM 实现为 TP=4 的本地投影形状补一份 H200 专用调度。

## 优化点

为模型 TP=4 的本地 `(N, K)` 形状增显式 H200 plans，覆盖 `M={1,2,4,8,16}` 而非复用 SM103 表；未显示可重复 H200 收益的 shape/M 点一律回退标准 linear。dispatch 仍严格限定 SM90、BF16、未量化层、packed row-major 输入与表中精确条目，SM103 行为不变，其他 shape/M/dtype/layout/量化方法/平台保留原路径。H200 调优方法：在 10 个投影 × 5 个 M 共 50 个 shape/M 点上，各搜 80-666 组合（block size、outputs per CTA、vector width、K unroll、动态/静态 K），决赛用 21 次 paired hot CUDA-graph replay 与 101 次 paired L2-flush replay，仅当两种 regime 的中位改进都 ≥约 3% 才保留，最终得 9 个形状共 31 个 plan，其余 19 点用标准 linear fallback。

## 关键改动

```text
vllm/models/qwen4_exp/nvidia/low_latency_gemm.py: 新增 QWEN4_EXP_SM90_GEMM_PLANS 调优表与窄门控的 SM90 dispatch
tests/kernels/test_bf16_skinny_gemm.py: 增 QWEN4_EXP_SM90_CASES，覆盖表完整性、routing 与 H200 正确性
```
```text
def test_qwen4_exp_hopper_plans_are_valid() -> None:
    plans = qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS

    assert len(plans) == 9
    assert sum(map(len, plans.values())) == 31
    assert (320, 10240) in plans
    assert (10240, 320) not in plans
```

## 性能声明

31 个独立验证点的 hot-replay speedup 为 1.096x 到 2.686x，最大绝对误差不超过 `0.015625`；HC-up `(10240, 320)` 在所有测试的 M 上都保持标准实现（fallback）。E2E serving A/B 以父提交 `e0d27040` 为 baseline，同一 4×H200、TP4、moe-backend triton、max-num-seqs 256、prefix caching、language-model-only、fastsafetensors，matched sequential decode（2 warmups + 24 measured，2048 in / 512 out，concurrency 1，temperature 0，ignore_eos，seed 401）：Output throughput 136.20 → 155.47 tok/s（+14.15%）、Request throughput 0.2660 → 0.3037 req/s（+14.15%）、Median TPOT 6.279 → 5.478 ms（-12.75%）、Mean TPOT 6.342 → 5.521 ms（-12.95%）、P99 TPOT 7.388 → 6.421 ms（-13.09%）、Median TTFT 447.87 → 406.89 ms（-9.15%，正文称该顺序 workload 下噪声大、不作稳健结论）；24 个请求中 23 个 TPOT 更低，平均 speedup 1.151x。测试条件：模型 Qwen3.8-Flash-Next、硬件 NVIDIA H200 SXM、精度 BF16、workload 为 2048/512 的 matched sequential decode。

## 验证与回归风险

`CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/kernels/test_bf16_skinny_gemm.py` → 215 passed、158 skipped、14 warnings；`pre-commit` 全 hooks 通过。H200 测试把 31 个 SM90 条目全部经生产 Qwen4Exp dispatch 执行，与 `torch.nn.functional.linear` 比 cosine similarity > 0.999。回归风险：dispatch 窄门控（SM90/BF16/未量化/packed row-major/精确表项），不支持的 shape/M/dtype/layout/量化/平台保留原路径，SM103 不变；TTFT 在该顺序 workload 有噪声，正文不作声明。正文注明用 OpenAI Codex 实现、测试与撰写，提交人负责复核。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
