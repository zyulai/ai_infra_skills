# RFC #27622: [RFC] Host-Backed Deferred Device KV Allocation for PD Disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27622](https://github.com/sgl-project/sglang/issues/27622) |
| 作者 | aooxin |
| 创建日期 | 2026-06-09 |
| 状态 | ⚪ closed (2026-08-09) |
| labels | (none) |
| 热度 | 👍 3 · 💬 1 |
| 分类 | Disaggregated Serving |

## 一句话结论
提出 PD direct-to-host 传输路径:KV 先暂存解码节点 host 内存、临近解码执行才分配显存,缩短非运行请求的显存驻留窗口。

## 提案动机
Decode 侧预分配在 prefill→decode KV 传输完成前就占用设备 KV。Agentic 长上下文 + 高 HiCache 命中率下,预分配需求随 prefill 输入吞吐而非实际可运行请求数扩张:大量未进入 running batch 的请求占住 decode 显存,active batch 远低于配置上限;显存紧张时 prealloc 队列排队推高 TTFT;staging 容量与执行容量紧耦合,使 KV 容量隐性控制计算并发。H20e 例:5M token KV 池对 100K 输入请求名义约 50 请求上下文,而生产 trace 显示约 33% 的 decode 侧设备 KV 驻留发生在进入 running batch 之前。

## 提出的方案
- 新增 decode-only 的 `--enable-pd-direct2host` 模式与解码节点 pinned host transit KV pool;decode 预分配改为分配 host KV slots
- 新路径:prefill 传输到 host 内存 → 传输完成后、H2D 就绪时才分配 device KV → 专用 CUDA stream 异步 H2D → 跨全部 TP rank 完成后才进入 waiting/running batch → 释放 host transit slots
- 配置:`--pd-direct2host-host-ratio 2.0` 或固定 `--pd-direct2host-host-size <GB>`;禁用时保留现有回退行为
- 约束:仅 decode 模式;初始实现与 decode KV offload、hierarchical cache 互斥;需足够 pinned host 内存;传输 backend 须支持注册并写 host buffer
- 部署注意:当前测试环境需 `export NCCL_GRAPH_MIXING_SUPPORT=0` 才能让 CUDA graph 与非 graph 执行可靠共存,视为初期部署注意项而非长期要求
- 明确边界:不减少活跃 decode 请求的最终显存需求,只缩短非运行请求占用显存的时间窗口

## 优化点与收益
正文未给出加速比数字;量化论据(照抄):H20e 案例中约 `33%` 的 decode 侧设备 KV 驻留发生在请求进入 active running batch 之前,有效预算 `5M * (1 - 0.33) = 3.35M tokens`,按 100K tokens/请求对应约 33 个活跃请求(名义 50 个)。预期效果:降低非运行请求显存驻留、提升长上下文 running batch size、缓解 prealloc 排队推高的 TTFT;代价是无拥塞场景多一步 H2D staging(可能提高 TTFT)。

## 现状与落地
已关闭(2026-08-09)。正文为设计提案,含 roadmap(原型、可观测指标、性能评测);落地情况需查关联 PR。

## 关联
正文未列出外部链接(仅 GitHub 附件示意图)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
