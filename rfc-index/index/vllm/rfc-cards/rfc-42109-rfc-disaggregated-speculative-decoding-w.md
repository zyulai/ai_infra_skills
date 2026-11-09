# RFC #42109: [RFC]: Disaggregated Speculative Decoding with Standalone Parallel Draft Model

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42109](https://github.com/vllm-project/vllm/issues/42109) |
| 作者 | laviier |
| 创建日期 | 2026-05-08 |
| 状态 | 🟢 open |
| labels | RFC, stale |
| 热度 | 👍 3 · 💬 5 |
| 分类 | Speculative Decoding |

## 一句话结论
提议把 draft 模型从 target TP 组拆到独立 GPU:推测缓存预建、draft/target 计算完全重叠,支持 N:M 扩展与更大的 draft 模型。

## 提案动机
当前 vLLM 的推测解码把 draft 与 target 共置:draft 跑在 target 的 TP 组上、抢同样的 SM/HBM 带宽、受 target 调度器驱动。两个结构性限制:1) draft 计算偷走 target 算力——TP=2 的 Llama-3-70B 高并发下,1B-8B 的 draft 即使算力空闲也会给 target 每步延迟增加 10-30%(共享 forward 关键路径);2) 缓存构建并行度被 target 调度卡死——draft 无法提前运行。

## 提出的方案
Disagg-SD 把 draft 移到专用 GPU(经 ZMQ + 共享主机或跨节点 RDMA):target 每步发验证结果,draft server 从预建的 speculation cache 直接返回 K 个草稿 token。三部分设计:
1. **独立 DraftServer 进程**(`python -m vllm.v1.spec_decode.draft_server`):自带 ModelRunner、独立于 target TP/调度器;ZMQ ROUTER(默认 `tcp://*:50051`),处理 PREFILL / SPECULATE / FREE_SEQ 三类命令;维护按 `(seq_id, k_accepted, bonus_token)` 键控的 SpeculationCache,支持超时驱逐;状态按 verify server 分区,防止跨租户缓存互相冲刷。
2. **Verify 侧 DisaggSpeculatorProxy**:驻留 target TP rank 0,持 DraftRouter(每 draft server 一个 ZmqDraftConnector),路由策略 round_robin(默认)/affinity;每步后发送 `(seq_ids, k_accepted, bonus_tokens)` 并阻塞等待(默认 5s 超时),不可用时优雅降级为零 token 草稿并周期重连;新增 `SpeculativeConfig` 字段(`disagg_draft_addresses`、`disagg_fan_out`、`disagg_jit_fallback` 等),全部由 `use_disagg()` 门控,不设置时与主线行为一致。
3. **几何 fan-out 推测缓存**:每个在途请求按接受位置 k∈[0,K] 预建 F_k 个 K-token 分支(依 SSD Theorem 12,k 越小接受概率越高、预算越多);缓存构建跑 tree decode(K 次 varlen 批量 forward,按分支隔离 KV 写);命中则换入专用 KV 块,未命中回退 JIT 推测保证延迟有界。

## 优化点与收益
- 内部 POC(已端到端跑通):N:M=3:2,Llama-3.1-70B(verify)+ Llama-3.2-1B(draft),在**相同总 GPU 预算**下与共置推测解码相比 **TPOT 改善约 1.4x**(作者内部延迟/吞吐基准)。
- 渐近分析(正文推导):共置时每轮 wall time 趋于 v + d,Disagg-SD 趋于 v,加速比 (v+d)/v = 1 + d/v;上限受缓存构建时间约束(d > v 且无法流水隐藏时失效)。
- 下一步:standalone 并行 draft 模型(mask-token MTP 式)单次 forward 算完所有 fan-out 分支,把 draft 侧延迟从 K × small_forward 降到 1 × medium_forward;POC 当前瓶颈是缓存构建(K=6、N=56 分支时为 launch-bound,每轮 6 个小 kernel)。

## 现状与落地
状态 open,反馈期四周。作者已有上述 Disagg-SD 框架的内部实现,并行 draft 模型仍在开发中;正文未提及上游 PR。

## 关联
正文未列出外部 RFC/PR/文档链接。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
