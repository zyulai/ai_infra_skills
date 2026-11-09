# PR #24271: [KDA] Optimize prefill kernels with diagonal and recompute fuse

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-05-09 |
| 分类 | Attention / Kernel |
| Label | run-ci |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 集中在 fla/ 下 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/24271 |

## 一句话结论

优化 KDA(Kimi Delta Attention)prefill 内核:对角块与 w/u/kg 重计算折入 inter+solve kernel、离轴 MMA 转 bf16、autotune 修网格利用率,TP=8 下 chunk_kda_fwd 单核最高提速约 25%。

## 动机

针对 `Kimi-Linear-48B-A3B-Instruct` 这类 hybrid linear-attention 模型在 Hopper GPU 上的 prefill 性能。Profiling 显示 KDA 占 prefill GPU 时间约 13%(4-GPU TP);两个主导 kernel(`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 与 `chunk_kda_fwd_kernel_inter_solve_fused`)的 autotune 配置受限,SM 利用率低下:前者的 grid `(V/BV, B·H_per_rank)` 在 TP=8、H=4 时只产生 16 个 CTA,远低于 Hopper 每 GPU 的 SM 数,大部分 SM 空转。

## 优化点

KDA 是 chunk 化 linear attention,每层前向分 4 个阶段(cumsum、chunk_intra 三子核、串行的 h-recurrence、输出 kernel)。改动分四条线:(1) h-recurrence kernel 的 autotune 扩大 BV 扫描到 16、num_warps 到 8,让小 per-rank H 时能选更宽的 grid;并加入 NT_BUCKET(NT ≤32 / ≤128 / >128 三档)到 autotune key,因为短 prefill 偏好大 BV、长 prefill 偏好小 BV。最终 diff 里该 kernel 实际收敛为单一定值 config(BV=32, num_warps=4, num_stages=2),NT_BUCKET 保留为前向兼容——注释说明多 config autotune 会因 kernel 原地写 initial_state 而在 benchmark 阶段污染 cache pool,restore_value 方案在生产规模下会 OOM。(2) chunk_intra 原本 3 个 kernel,两处 HBM 中间往返(Akkd、Akk_inv);新增 `FUSE_DIAGONAL`(对角 Akk 块驻留寄存器直接供 forward substitution 使用)与 `FUSE_RECOMPUTE`(solve 后 Akk_inv 子块留在寄存器,立即构造 w=β·k·Akk⁻¹、u=β·v·Akk⁻¹、kg=k·exp(gn−gk))两个 constexpr 模式,消除中间张量。(3) 离轴 `tl.dot` 的输入(`b_q·b_gqn`、`b_k·b_gqn`、key tile)从 fp32 转 bf16 后做 MMA,让 Hopper tensor core 跑 bf16 速率(约 2x 吞吐),accumulator 保持 fp32。(4) kda.py dispatcher 按 `_small_grid = B·NT_pr·H_per_rank ≤ 256` 决定是否开融合:小 grid 受 launch 开销主导时融合,大 grid 时寄存器溢出则回退非融合;varlen 下 NT_pr 用 `chunk_indices.shape[0]` 正确计数。另外 `chunk_gla_fwd_kernel_o` 的 `tl.dot` 去掉 `allow_tf32=False`,`recompute_w_u_fwd_kernel` 的 DOT_PRECISION 改为 tf32 并扩大 autotune,`cumsum.py` 的 autotune 加入 num_stages 维度。

## 关键改动

```text
python/sglang/srt/layers/attention/fla/chunk_intra.py: FUSE_DIAGONAL/FUSE_RECOMPUTE 两模式折入 inter_solve_fused,离轴 MMA 转 bf16
python/sglang/srt/layers/attention/fla/chunk_delta_h.py: h-recurrence autotune 收敛为单 config,NT_BUCKET 进 key
python/sglang/srt/layers/attention/fla/kda.py: grid-aware 融合开关 _small_grid;recompute/gla_o kernel 配置调整
python/sglang/srt/layers/attention/fla/cumsum.py: autotune 增加 num_stages 扫描
```

融合开关的判据:

```python
_small_grid = _B * _NT_pr * _H_pr <= 256
...
chunk_kda_fwd_intra(..., fuse_diagonal=_small_grid, fuse_recompute=_small_grid)
```

## 性能声明

测试条件均针对 Kimi-Linear-48B-A3B-Instruct、attention-backend=triton。

Profile( prefetch B=1 T=8192,E2E prefill per-kernel GPU time):`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 5.00 → 3.30 ms(−34%),`chunk_kda_fwd_kernel_inter_solve_fused` 3.03 → 1.84 ms(−39%),KDA total 12.14 → 9.37 ms(−23%);`chunk_gla_fwd_kernel_o` 反而 +17%(1.03 → 1.21 ms)。

孤立 kernel benchmark(`chunk_kda_fwd`,ms/iter,200 iters,20 warmup,K=V=128,BT=64):TP=8(H=4/rank)下 T=2048 为 0.285 → 0.216(+24%)、T=4096 +24%、T=8192 +9%、T=16384 +12%、varlen 4×4096 +25%;TP=4(H=8)下 T=2048 −24%,T=8192/16384 约 −2%。

精度:GSM8K 90.0%(200 题,8-shot,无回归);`test_chunk_gated_delta_rule.py` 28/28 PASS(ATOL=2e-2);融合测试(test_fuse_diagonal / test_fuse_recompute / test_fused_ho)对 PyTorch 参考实现 max-diff 在 3×10⁻⁵ – 3×10⁻³ 内。

## 验证与回归风险

依赖注册测试 `test/registered/attention/test_chunk_gated_delta_rule.py`(28 用例)与 `test/registered/models/test_kimi_linear_models.py::TestKimiLinear::test_gsm8k`。风险点:精度上 bf16 MMA 与 tf32 dot 改变数值路径(用 ATOL=2e-2 与 GSM8K 验证,理论 diff 上限 3×10⁻³);`chunk_gla_fwd_kernel_o` 反而变慢(+17%),autotune 收敛到 BK=64/BV=64 与 triton 3.6 的兼容性敏感(更宽 sweep 会 crash 或选次优);h-recurrence kernel 原地写 initial_state 的 autotune 约束是硬性设计限制,若未来加 config 需先重构输出 buffer;融合模式在小 grid 生效,大 grid 行为不变,但 varlen chunk 计数逻辑变化影响所有 IS_VARLEN 路径。

## 关联

同区域优化:fla(Kimi linear attention)系列 kernel PR 与 Kimi-Linear 模型支持 PR,同属 Attention / Kernel 分类。PR 正文未给出具体关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
