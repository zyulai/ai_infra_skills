# RFC #48117: [RFC]: [ROCm][DistInf]: Distributed Inference (Disaggregated P/D + Wide-EP) CI Enablement on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48117](https://github.com/vllm-project/vllm/issues/48117) |
| 作者 | lcskrishna |
| 创建日期 | 2026-07-09 |
| 状态 | 🟢 open |
| labels | RFC, rocm, unstale |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Disaggregated Serving |

## 一句话结论
提议在 ROCm 上建立专门的分布式推理 CI 流水线,持续验证 disaggregated P/D 与 Wide-EP,并以端到端 GSM8K 精度做门禁。

## 提案动机
分布式/分解推理已成为 vLLM 一等 workload,但上游没有 CI 持续验证 ROCm 上的 disaggregated P/D 或 Wide-EP:ROCm 分解路径的回归(PD 精度、KV transfer、MoRI-IO connector、expert-parallel all2all)只能手动且滞后发现;多节点/RDMA 代码路径(prefill 经 RoCE NIC 向 decode 推 KV)是栈中最脆弱的部分却零自动化覆盖;分解路径没有端到端精度(如 GSM8K)门禁,精度漂移无法察觉。

## 提出的方案
- 在 .buildkite/amd-disagg/ 下建自包含 DI CI harness(PR #47030),经 Buildkite + Slurm 跑在多节点 AMD 集群上
- 拉起完整 P/D 拓扑(proxy/router + prefill + decode),每个 server 先过健康门禁再发流量;跑端到端 GSM8K 精度检查(可选 benchmark 模式)
- 两种模式:WIDE_EP_MODE=0 一般 P/D(独立 TP8 server);WIDE_EP_MODE=1 DP + Expert Parallelism(如 DP8/EP8/TP1,及 2P2D DP16/EP)
- 组件:vllm_disagg.sh、cluster.sh、run-slurm-disagg-test.sh、run_xPyD_disagg.slurm、pipeline.disagg.yaml、models.yaml
- 测试矩阵:Stage 1 为 1P1D(TP8 与 Wide-EP DP8/EP8/TP1),MoRI-IO connector + toy proxy/vllm-router,GSM8K 门禁;Stage 2 加 2P2D 多 PD;Stage 3 加 2P2D DP16/EP;初始目标模型 DSV3
- 分阶段推进:Stage 1 非门禁 nightly(依赖 PR 46482、47764),Stage 2 扩大矩阵,Stage 3 Wide-EP 特性

## 优化点与收益
正文未给出量化数字。本 RFC 属 CI/质量保障建设,收益是 ROCm 分解路径回归更早发现、多节点/RDMA 路径获得自动化覆盖、精度有门禁。

## 现状与落地
🟢 open;初始实现见 PR #47030,RFC 用于让社区对齐更广的计划(架构、测试矩阵、分阶段 rollout、infra 需求与开放设计问题)后再扩展覆盖并转为门禁;评论 9 条。

## 关联
- PR #47030(初始实现)、PR #46482、PR #47764

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
