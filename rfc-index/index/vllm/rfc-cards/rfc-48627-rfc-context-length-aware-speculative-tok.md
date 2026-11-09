# RFC #48627: [RFC]: Context-length-aware speculative token scheduling — extending num_speculative_tokens_per_batch_size with a context-length axis

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48627](https://github.com/vllm-project/vllm/issues/48627) |
| 作者 | seongyun1104 |
| 创建日期 | 2026-07-14 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 2 · 💬 8 |
| 分类 | Speculative Decoding |

## 一句话结论
提议给推测深度调度表 `num_speculative_tokens_per_batch_size` 增加可选的上下文长度维度,使 K 由 (batch, ctx) 二维表决定。

## 提案动机
现有配置按 batch size 调度推测深度 K;"高 batch 下推测不再划算"这条经验法则,作者认为是短上下文场景的产物而非普遍规律。作者在 Gemma-4-31B(FP8,混合滑动+全局注意力)+ MTP drafter、单卡 H100 NVL 94GB、vLLM 0.23/0.24、greedy、120s 稳态窗口的测量中观察到:短上下文高并发下 K 增益收敛(复现既有 crossover),而同样高 batch、上下文增长时(~98% APC 命中)增益重新出现。机制上,长上下文 decode 受每步 KV 读的内存带宽约束,验证 K 个草稿 token 可把这次读摊销到 K+1 个 token 上。这最契合 agentic/RAG/多轮等长共享前缀高并发负载。

## 提出的方案
- **Schema**:在 `vllm/config/speculative.py` 与 `v1/spec_decode/dynamic/utils.py` 中接受 5 元组 `(bs_lo, bs_hi, ctx_lo, ctx_hi, K)`,与现有 3 元组并存(3 元组解释为覆盖全部上下文长度),向后兼容;沿用既有校验规则(闭区间、bs 从 1 全覆盖、不重叠、每 bs 段 ctx 全覆盖)。
- **运行时查找**:`v1/core/sched/scheduler.py` 的查找扩展为 `dense[B][ctx_bucket]`;上下文代表值取已调度请求 `num_computed_tokens` 的归约(实现默认 decode-only 池的整数均值,归约方式选择未经充分验证,成本未测量);`SchedulerOutput.num_spec_tokens_to_schedule` 保持每步标量不变。
- **图与缓冲**:CUDA graph 与 buffer 按 K 值而非表单元数索引,设计预期是相同 K palette 下 2D 表与 1D 表捕获同样的图集;正文已撤回"ctx 轴零资源开销"的结论,每步归约是真实开销且未测量。
- **文档要点**:表索引是每步已调度请求数而非客户端并发,分层边界应放在运行分布尾部之外;摊销收益在 ~2k 附近饱和,长尾用一个"≳2k"桶即可;K=0 档并不免费,开销集中在 TTFT(p50 0.28s → 2.0s)而非 TPOT(+6.7%)。
- **定位**:三层策略阶梯的第 1 层(声明式 (bs, ctx) 表,零延迟先验);第 2 层按桶接受率修正、第 3 层成本模型×接受率(#47808、#48692 的 DSpark 方向)均可叠加,消费同一调度面。

## 优化点与收益
原始观察(同上 Gemma-4-31B/H100 条件,client concurrency 256):ctx ~460/970/1990/4096 时 K=3 对 K=0 吞吐为 1.17x/1.25x/1.38x/1.36x,K=3 TPOT 稳定在 ~78 ms 而 K=0 TPOT 爬升(109→138 ms)。
**但作者已于 2026-08-31 与 2026-10-22 两次更正证据**:原表带预热偏差;更严格方法学下推测 vs 无推测为 0.79x/0.93x/1.00x/1.10x(ctx 400/900/1900/4000),在 ctx≈1900 附近才过盈亏点;与 static K=3 相比差异为 −11.64%/+1.94%/+1.42%/−0.73%,未确立优势。仍然成立的是:(batch × ctx) 调度对 batch-only 调度在 ctx 900/1900/4000 为 1.29x/1.30x/1.36x(c=256,预注册实验)——即上下文轴调度优于 batch-only 调度,但对 static K=3 的优势未确立,性能论证已撤回。

## 现状与落地
状态 open,反馈期 2 周。作者曾实现 PR #48944,2026-10-22 重分析后**该 PR 暂停、待进一步验证**;核心研究问题("同 batch 下最优 K 是否随上下文显著不同")保持开放。测量工具、协议与 EVIDENCE_LEDGER.md 公开于 https://github.com/seongyun1104/depthchart。已知局限:单一模型/drafter 组合、仅前缀缓存命中场景、单卡 TP=1。

## 关联
- PR #48944(实现 PR,现已 on hold)
- https://github.com/seongyun1104/depthchart(测量 harness 与证据台账)
- MagicDec(arXiv:2408.11049)与 TurboSpec/SmartSpec(arXiv:2406.14066)
- #47277、#48692、#47808(相关讨论与自适应验证方向)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
