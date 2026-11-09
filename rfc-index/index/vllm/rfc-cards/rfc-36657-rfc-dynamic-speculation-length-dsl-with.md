# RFC #36657: [RFC]: Dynamic Speculation Length (DSL) with Confidence-Threshold Early Exit for vLLM Speculative Decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36657](https://github.com/vllm-project/vllm/issues/36657) |
| 作者 | jmamou |
| 创建日期 | 2026-03-10 |
| 状态 | ⚪ closed(关闭于 2026-07-27) |
| labels | RFC, stale |
| 热度 | 👍 2 · 💬 5 |
| 分类 | Speculative Decoding |

## 一句话结论
提议动态推测长度(DSL):草稿步置信度低于阈值即提前退出,自适应草稿长度减少无效计算、提升吞吐。

## 提案动机
标准 speculative decoding 每次解码迭代固定让草稿模型跑 `num_speculative_tokens` 步;草稿链前段置信度低时,后续步几乎必然被目标模型拒绝,草稿 forward 是纯浪费。设得保守(如 5)在高置信序列上损失吞吐,设得激进(如 20)在低置信序列上浪费算力。DSL 受 [DISCO (Mamou et al., 2024)](https://arxiv.org/abs/2405.04304) 启发,在运行时自适应草稿长度来化解这一矛盾。

## 提出的方案
- 在 `SpeculativeConfig` 新增 `draft_confidence_threshold` 字段(默认 `0.0` 即关闭);启用后 `DraftModelProposer` 在每个草稿步 i 评估 batch 级均值退出条件 (1/B)·Σ_j max_v softmax(z_i,j) < τ,条件成立即提前停止草稿、返回已生成 token。
- 输出张量零填充到完整 `num_speculative_tokens` 宽度,保持固定形状,下游(GDN attention、scheduler、KV-cache)无感知;零填充位置由 verifier 拒绝,不影响输出质量。
- Mean policy:退出条件用 batch 均值置信度而非最小值,避免混合批次中单个不确定请求打断整个批次的草稿。
- 无接口变化:`propose()` 始终返回 `[batch_size, num_speculative_tokens]`。
- 新增 DSL 指标(`dsl_num_proposals`、`dsl_num_early_exits`、`dsl_tokens_generated`、`dsl_tokens_requested`),从草稿模型经 `ModelRunnerOutput`、scheduler 到 benchmark 输出端到端传播。
- 实现见关联 PR #35301;用法:`speculative_config` 中设 `num_speculative_tokens: 20`、`draft_confidence_threshold: 0.6`,或 CLI `--draft-confidence-threshold 0.6`。

## 优化点与收益
正文给出基准数字,测试硬件为 1× A100 80 GB PCIe:

| 模型对 | 数据集 | 配置 | vs. target | vs. SD/5 |
|---|---|---|---|---|
| Llama-3.1-8B + Llama-3.2-1B | mt-bench c=1 | DSL/20/τ=0.6 | **1.77×** | **1.44×** |
| OPT-6.7B + OPT-125M | mt-bench c=1 | DSL/10/τ=0.6 | **3.10×** | **1.64×** |
| Llama-3.1-8B + Llama-3.2-1B | random c=8 | DSL/15/τ=0.8 | **3.60×** | **1.52×** |
| Llama-3.1-8B + Llama-3.2-1B | mt-bench c=4 | DSL/20/τ=0.4 | **1.32×** | **1.63×** |

## 现状与落地
已关闭(关闭于 2026-07-27),labels 含 stale。正文列出实现 PR #35301,feedback period 至 2026-03-17,但未写明 PR 是否已合并;落地情况需查关联 PR。正文同时写明已知局限:early-exit 检查对 GPU 标量调 `.item()` 会在每个草稿步强制 host–device sync,小 batch 下可忽略、高并发下可能成瓶颈,device 侧免同步方案留作后续;阈值不自动调优,推荐起始范围 `0.4–0.6`,最优值依赖模型对与数据集;本 PR 范围内仅 `DraftModelProposer` 支持 DSL(EAGLE/EAGLE2 等 proposer 不受影响)。

## 关联
- PR: https://github.com/vllm-project/vllm/pull/35301
- DISCO 论文: https://arxiv.org/abs/2405.04304

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
