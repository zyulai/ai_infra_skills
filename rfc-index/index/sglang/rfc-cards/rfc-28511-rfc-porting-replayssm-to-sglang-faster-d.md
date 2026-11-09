# RFC #28511: [RFC] Porting ReplaySSM to SGLang: faster decode and speculative decoding for hybrid (GDN/KDA) models

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28511](https://github.com/sgl-project/sglang/issues/28511) |
| 作者 | yuan-luo |
| 创建日期 | 2026-06-17 |
| 状态 | ⚪ closed (2026-07-21) |
| labels | RFC, linear-attention |
| 热度 | 👍 8 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
移植 ReplaySSM:把 GDN/KDA 解码的每步全状态读写改为「读为主 + 周期 flush」,省带宽提速解码与投机解码。

## 提案动机
GDN(Gated DeltaNet,Qwen3.5/Qwen3-Next 一脉)与 KDA 解码每步从 HBM 读固定尺寸循环状态 S(每 slot 每层约 1 MB bf16),做 rank-1 更新后写回,每步约 2|S| 状态流量。该 kernel 在服务 batch 下是 memory-bound:H20-3e(峰值拷贝带宽约 3.8 TB/s)上实测 batch 64/256 的 HBM 带宽利用率分别为 70%/79%(batch 16 为 29%,batch 1 为 2%)——此时减字节几乎等于减运行时间。

## 提出的方案
- 核心思路:冻结 checkpoint S0,仅保留最近 L 步 (d,k,g) 的 per-slot ring buffer;非 flush 步只追加记录,用 S0+ring 以精确代数恒等式重建读出,不在 HBM 物化完整状态矩阵((k_j^T q) 先算成标量);每 L 步 flush 一次把 ring 折入 S0。状态更新对 S 线性,重建用小 GEMM 跑 tensor cores。
- Part A(已实现,#28451):缓冲 output-only GDN 解码 kernel + 完整集成——ring 内存池与 write_pos 生命周期、CUDA-graph 安全(捕获外推进)、radix force_flush 与快照条件逐字节对齐、COW 时 ring 重置;服务端参数 --enable-gdn-replayssm、--gdn-replayssm-cache-len(默认 16)。
- Part B(已实现,#28695):spec-decode verify 的 chunked 环形重放,把全状态写从每 verify 步摊销到每 L 个 committed token,拒绝回滚变成 ring 上的指针移动;前置决策门:实测目标 batch 下 verify 的 HBM 带宽(≥~70% 才值得做,<~30% 就不做)。
- 关联项 #27658(紧凑线性 spec cache,in review):已解决显存半边(存紧凑 (k, Δv, decay) 而非完整中间状态,2.32 GB → 0.06 GB,并明确报告吞吐无变化);Part B 与其互补而非替代。
- 精度立场:数学上无损但浮点非逐位一致(TF32/bf16 tensor cores 重结合);强行 IEEE 精度会禁用 tensor cores、慢 10–20×,位精确与提速互斥。

## 优化点与收益
- kernel microbench:batch ≥ 16、L=8 时 1.2–1.5×;batch 1 略慢(overhead-bound,非工作点)。
- 端到端:Qwen3.5-35B-A3B、128 并发下 TPOT 改善 ~2.3%(MoE 稀释了 GDN 收益,GDN-dense 模型收益更大)。
- 理论上限:L=8 时约 0.57× 流量 → ~1.75×;实际 1.2–1.5× 受带宽利用率、每步仍读 S0、重建计算与 ring 流量影响。
- 验证:GSM8K(radix + CUDA-graph on)flag-on 0.900 vs baseline 0.850,噪声内无回归;L=1 与基线逐位一致(atol 2e-6)。

## 现状与落地
已关闭(2026-07-21)。任务清单显示 Part A(#28451)、决策门与 Part B(#28695)均已完成;KDA decode 覆盖为未完成 follow-up(另有 bf16/fp8 ring、自适应 L 等可能后续)。

## 关联
- ReplaySSM 博客(Dao AI Lab): https://dao-lab.ai/blog/2026/replayssm/
- PR #28451(Part A): https://github.com/sgl-project/sglang/pull/28451
- PR #28695(Part B): https://github.com/sgl-project/sglang/pull/28695
- #27658(紧凑 spec cache): https://github.com/sgl-project/sglang/issues/27658
- Marconi(arXiv 2411.19379): https://arxiv.org/abs/2411.19379

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
