# RFC #46107: [RFC]: Heterogeneous TP<->DP parallelism across the P/D disaggregation boundary (MoRIIO)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46107](https://github.com/vllm-project/vllm/issues/46107) |
| 作者 | edwinlim0919 |
| 创建日期 | 2026-06-18 |
| 状态 | ⚪ closed(关闭于 2026-08-31) |
| labels | rocm |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议 MoRIIO 支持跨 P/D 边界的异构并行(P_TP8 配 D_DP8EP):prefill 用 TP 拿低 TTFT,decode 用 DP/EP 拿可扩展吞吐。

## 提案动机
分离式 prefill/decode 本允许两阶段各用最优并行度,但 MoRIIO 今天假设 prefill 与 decode 引擎共享同一并行布局(DP8EP↔DP8EP 或 TP8↔TP8)。两阶段实际受益于不同配置:在 8K-in/1K-out DeepSeek-R1 workload(MI300X / gfx942)上,prefill 是 latency-bound → TP 赢 TTFT(TP8 ≈ 1s vs DP8EP ≈ 10s);decode 在较高并发下是 throughput-bound → DP/EP 持续扩展(DP8EP 在 ≥ c256 仍可扩展)。因此高并发下最优部署可能是异构的:prefill 节点配 TP8、decode 节点配 DP8EP,同时获得 TP8 prefill 的低 TTFT 与高请求负载(如 512 并发)下的可扩展吞吐。

## 提出的方案
P/D KV transfer 需要在 `(dp_size, tp_size)` 不同的引擎间路由读请求,设计要点:
- **源 rank 解析放在 connector 而非 proxy**:proxy 只做粗粒度 instance + DP-rank 请求分发;KV 源 rank 由 connector 决定,因为只有 connector 知道 KV 布局(MLA-replicated vs partitioned)。
- **按 `(dp, tp)` 键控 session**:session/handshake 状态按 `(engine, dp_rank, tp_rank)` 键控(`get_engine_name_with_dp_tp`),每个远端 prefill rank 拥有各自的 session 与 memory-region metadata,而非单一 DP-only key。
- **Flexible prefill-TP read(本 RFC 旗标机制)**:TP-replicated prefill(prefill TP + MLA)喂 TP1 decode 时,latent KV 在所有 prefill TP rank 上复制,任何 rank 都是有效源;过去异构 PD 只从单一 TP rank(TP0)读,现由 connector 以确定性 round-robin 把读与 prefill NIC 负载摊到 TP0..N-1。门控:`world_size==1`、`use_mla`(KV 复制)、`remote_dp_size==1 && remote_tp_size>1`(纯 TP prefill 且多 rank);门外行为字节级不变。Future work:intra-node DP×TP prefill(如 DP4TP2)暂回退固定 rank 读,扩展到"请求所属 DP partition 内随机 TP rank"是局部后续工作。
- **一致的 rank 贯穿**:所选 rank 流经 session key、build-on-demand handshake、peer-host 解析与读完成 notify port,保证读与其完成通知始终寻址同一 prefill rank(错位会泄漏 prefill buffer → MR overflow)。
- 实现拆为三个相互独立、均 standalone 于 `main` 的 PR:#46116(Core/KV-transfer:DP-rank 读路由、per-(dp,tp) session keying、flexible prefill-TP read、eager pre-forward handshake 防混合配置死锁)、#46115(Bugfix:toy-proxy DP-rank index-aliasing 修复 + 高并发 proxy 加固)、#47495(Bugfix:RDMA send-queue 满时的 backpressure 重试)。
- 已启用并验证的目标拓扑:Forward `P_DP8EP → D_TP8`(即 DP/EP prefill、TP decode);镜像 `P_TP8 → D_DP8EP` 因评估中性能不佳未做广泛验证,列为 future work。

## 优化点与收益
正文照抄数字(8K-in/1K-out DeepSeek-R1,MI300X / gfx942;每并发级发送 10×C 个请求并预热)。c512 时:

| 配置 | Total tok/s | Generation tok/s | Mean TPOT | Mean TTFT |
|---|---|---|---|---|
| 1: 2P_TP8 : 1D_TP8(同构 TP8) | 38,407 | 4,263 | 55.9 ms | 55,179 ms |
| 2: 2P_DP8EP : 1D_DP8EP(同构 DP8EP) | 34,935 | 3,878 | 42.3 ms | 77,155 ms |
| 3: 2P_TP8 : 1D_DP8EP(MIXED) | **42,349** | **4,701** | 55.0 ms | **46,052 ms** |

所测最高并发(512 并发)下,混合并行 2P_TP8:1D_DP8EP 取得三配置中最高吞吐与最低 TTFT(基准耗时也最短,1,003.2 s);prefill 阶段 TP8 ≈ 1s vs DP8EP ≈ 10s。

## 现状与落地
已关闭(关闭于 2026-08-31)。正文写明 Forward 拓扑(P_DP8EP → D_TP8)"已被本 RFC 的 PR 启用并验证",三个实现 PR 均声称 standalone on `main`(#46116 早期草稿曾叠在 in-flight wideTP PRs #45228/#45230 上,后已重建为独立);镜像拓扑留作 future work。PR 是否已合并正文未写明,落地情况需查关联 PR。注:本 RFC 为 ROCm/MoRIIO 特定(gfx942、AITER MLA)。

## 关联
- 实现 PR: https://github.com/vllm-project/vllm/pull/46116
- 实现 PR: https://github.com/vllm-project/vllm/pull/46115
- 实现 PR: https://github.com/vllm-project/vllm/pull/47495
- 早期依赖的 wideTP PRs: #45228 / #45230
- 最近 prior art(NIXL block-size,不同轴): #42895 / RFC #26744 / PR #30275

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
