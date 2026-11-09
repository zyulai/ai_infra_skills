# PR #32340: [AMD][DSV4]perf: shared experts fusion top6

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @karverma-amd |
| 合并 | 2026-08-20 |
| 分类 | Model-specific Fast Path |
| Label | amd, deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/32340 |

## 一句话结论

修复 DeepSeek-V4(top-6 路由)在 MI355X 上 shared experts fusion 的启动崩溃,低并发输出吞吐 +8-11%。

## 动机

DeepSeek-V4 在 MI355X(gfx950)上开启 `--enforce-shared-experts-fusion` 直接启动崩溃:fused topk/append 路径沿用了 DeepSeek-V3 的两个约定——fp32 correction bias 与 2 的幂次 topk,而 DSV4 是 bf16 correction bias 加 top-6 路由,分别触发 `AssertionError: bias must be float32` 与 Triton `arange's range must be a power of 2`。修复后 fused 路径可跑通,并验证为低并发下的吞吐收益。

## 优化点

三处机制修复:(1) `moe_fused_gate` 放宽 dtype 断言,`_router_triton_kernel` 在 load 时把 bias 上转型 fp32(bf16/fp16→fp32 是精确加宽),`topk.py` 相应去掉每次 MoE 调用对静态 routing bias 做 host 端 fp32 拷贝的开销;(2) `_fused_append_shared_experts_kernel` 与 `_fused_append_remap_shared_experts_deepep_kernel` 改用 `next_power_of_2(K/S)` 块加掩码迭代,支持非 2 幂的 topk(如 K=6)与共享专家数(如 S=3),对 2 幂 K 行为不变;(3) 修正 DeepEP interleaved remap 公式:`e -> e + (e // num_local_routed) * S`,旧的 `e + e // num_local_routed` 在 S>1 时会让 routed id 与前一 rank 的 shared slot 冲突,S==1 时退化为旧公式。

## 关键改动

```text
python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py: 两个 append kernel 增加 BLOCK_K/BLOCK_S 掩码迭代支持非 2 幂
python/sglang/kernels/ops/moe/moe_fused_gate.py: bias 接受 fp32/fp16/bf16,kernel 内上转型
python/sglang/srt/layers/moe/topk.py: 直接传 bf16 bias;remap 公式改为按 S 宽度插 gap
test/registered/moe/test_fused_append_shared_experts_top6.py: 新增非 2 幂 K/S 单测(K=6 用例)
```

关键公式修正:

```text
# DeepEP interleaved layout: shift each routed id past ALL shared slots
# that precede its rank; S == 1 reduces to the old formula.
- ids = ids + ids // num_local_routed
+ ids = ids + (ids // num_local_routed) * S
```

## 性能声明

测试条件:deepseek-ai/DeepSeek-V4-Pro,MI355X(gfx950)TP4,image `rocm/sgl-dev:v0.5.15.post1-rocm720-mi35x-20260724`;ISL=8192,OSL=1024,output token throughput,fusion vs baseline:

| 并发 | baseline (tok/s) | fusion (tok/s) | Δ |
|---|---|---|---|
| 1 | 69.59 | 77.06 | +10.7% |
| 2 | 122.31 | 135.33 | +10.6% |
| 4 | 216.26 | 238.91 | +10.5% |
| 8 | 366.29 | 403.37 | +10.1% |
| 16 | 569.47 | 613.60 | +7.7% |

精度:GSM8K 2000 题 × 3 轮,baseline 平均 0.939,fusion 平均 0.940,Invalid ≈ 0.000,在 run-to-run 噪声内持平。

## 验证与回归风险

新增 `test_fused_append_shared_experts_top6.py`(对 pure-torch golden reference,覆盖非 2 幂 K/S,K=8 作为 2 幂回归守卫;PR 报告 3 passed, 7 subtests passed);`test_moe_fused_gate.py` 新增非 fp32 bias 与 fp32 bias 路由逐位一致(atol=0)的测试;`test_fused_append_remap_per_rank_shared_slots.py` 新增 S>1 时 routed/shared id 无冲突的回归守卫。风险面:改动位于所有 MoE 模型共用的 fused gate 与 append kernel,DeepSeek-V3 等 2 幂 topk 模型路径行为应不变(有回归守卫),但 bias dtype 放宽与 remap 公式变更影响所有走 `moe_fused_gate`/DeepEP remap 的 EP 部署;PR CI 显示 Base 通过、Extra 失败(x)。

## 关联

AMD + MoE/DSV4 区域:PR #37889([AMD] GLM DSA prefill top-k v2 kernel)、PR #34837([AMD] K3 aiter prefill kernel)。上游参考 chitu 无;本 PR 同时是 fusion 路径在非 V3 拓扑上的首次可用化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
