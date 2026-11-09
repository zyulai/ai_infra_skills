# RFC #27080: [RFC]: To Inductor partition or to not Inductor partition (by default in v0.11.1)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27080](https://github.com/vllm-project/vllm/issues/27080) |
| 作者 | ProExpertProg |
| 创建日期 | 2025-10-17 |
| 状态 | ⚪ closed(关闭于 2026-06-20) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 10 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
讨论 v0.11.1 是否默认启用 Inductor partition:换 2-10% 性能提升,代价是编译时间增加与稳定性风险。

## 提案动机
vLLM main 上 piecewise cudagraphs 目前靠 Dynamo partitioning(在图进入 Inductor 前切分 fx.Graph)实现,这与需要看到整图的编译 pass 不兼容——如 attention+quant fusion 与 sequence parallelism(及 async tp),而除去 allreduce+rmsnorm(+quant) fusion 外,这些正是收益最大的 pass。vLLM 与 torch.compile 的合作产出了一套自定义 Inductor partitioning 方案(RFC #23261,PR #24281):本身就能通过降低 cudagraph replay 开销带来更好性能,并让 piecewise cudagraphs 与 fullgraph 自定义 pass 兼容;但它要求 torch==2.9,需要若干 monkeypatch 与 workaround(如 #26735、#26878、#26956、#26952),显著增加首次冷启动编译时间,且仍是偏实验性的特性(已知问题大多已解决,如 #26988 attention-nvfp4 fusion + inductor partition 的 Inductor codegen 错误,及若干 CI 问题 #26738)。因此需要在 0.11.1 发布前决定是否默认开启。

## 提出的方案
- 核心决策:是否在 0.11.1 默认启用 Inductor partition——上侧是性能,下侧是破坏风险与编译时间。
- 若默认启用,还有第二层问题:是否默认启用 attention+quant fusion(额外最多 2-6%)与 SP+AsyncTP(数字 TBD);二者均依赖 Inductor partition。由于未跨硬件广泛基准测试(如 attention+quant fusion 在 Blackwell + torch==2.9 上使 llama-70B tp=4 变慢,原因未知),大概率不默认开启这两个。
- 用户始终可手动关闭;随优化等级方案(RFC #20283, PR #26847)落地,`-O1` 即可轻松切换。
- 备选:先以 `-O1`(FULL_AND_PIECEWISE cudagraphs)为默认,用户用 `-O2` 选择性开启 inductor partition 与编译 pass,下个版本再把默认从 `-O1` 切到 `-O2`(作者表示若不默认开 inductor partition,这是其偏好方案)。
- 优化等级终态附录:`-O0` 不编译快启动(近似 --enforce-eager);`-O1` Dynamo partition + PIECEWISE cudagraphs + 简单 pass(SiluMul+quant fusion);`-O2` 全量优化(Inductor partition + FULL_AND_PIECEWISE + 全部 pass);`-O3` 未来做 max autotune。

## 优化点与收益
正文给出数字(条件:torch 2.9 + #24604;H100 SP-AsyncTP 另用 #26975):
- 性能收益:不同模型与 QPS 场景下 **2-10%**。
- attention+quant fusion 额外收益:最多 **2-6%**。
- 冷启动编译时间:**约 2-5x**(依模型而定);warm start 反而略快(需加载的产物更少)。作者自述见过大模型冷启动编译长达 100s。
- 基准硬件/模型:1xB200 llama-3.1-8B-Instruct-FP8、4xB200 llama-3.1-70B-Instruct-FP8、4xB200 Llama-4-Scout-17B-16E-Instruct-FP8、4xH100 llama-3.1-70B-Instruct-FP8(详细数值以原帖附图为准)。

## 现状与落地
已关闭(2026-06-20)。反馈期到 0.11.1 发布(2025-10-17 周五晚或周末);正文是征求默认开关决策的讨论,最终采纳哪条路线从正文无法看出,落地情况需查关联 PR 与发布说明。

## 关联
- [RFC #23261](https://github.com/vllm-project/vllm/issues/23261)(自定义 Inductor partitioning 方案)
- [PR #24281](https://github.com/vllm-project/vllm/pull/24281)(对应实现)
- [RFC #20283](https://github.com/vllm-project/vllm/issues/20283) / [PR #26847](https://github.com/vllm-project/vllm/pull/26847)(优化等级)
- [PR #24604](https://github.com/vllm-project/vllm/pull/24604)(基准数据所含改动)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
