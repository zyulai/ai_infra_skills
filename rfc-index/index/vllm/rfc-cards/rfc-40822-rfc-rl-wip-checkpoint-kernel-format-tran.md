# RFC #40822: [RFC]: [RL - WIP] Checkpoint -> Kernel Format transition graph

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40822](https://github.com/vllm-project/vllm/issues/40822) |
| 作者 | S1ro1 |
| 创建日期 | 2026-04-24 |
| 状态 | ⚪ closed(关闭于 2026-08-28) |
| labels | RFC, stale |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
提议文档化 checkpoint→kernel 格式转换图,配合 RDMA PUSH 式权重传输,消除 RL 权重同步中的重复后处理与 NCCL 瓶颈。

## 提案动机
RL 工作负载中的权重传输通常用单次 NCCL broadcast:每个 rank 通过 iterator 加载全量模型权重并各自执行全部后处理——浪费且可能成为瓶颈;它要求创建全局 process group、大量同步,且最多利用 1 张 NIC(50GB/s)。改用 RDMA 类传输(IB/EFA 等)则几乎免同步、直接读写远端 GPU 内存:trainer 与 inference 交换 metadata 后,由 trainer push 或 inference 引擎 pull 权重,可显著加速。但难点变成:如何知道权重在 inference/trainer 内存中的布局。权重从 HF 类 checkpoint 经处理进入 kernel format(如量化为 FP8 并把 scale 重排成 TMA 友好格式、`qkv`/`gate_up` 融合、TP 的 `ColumnLinear`),而该格式依赖运行时配置——例如跑 GLM5 FP8 时,用 DeepGemm 还是 Cutlass FP8 MoE,gate-up 投影可能排成 `w1w3` 或 `w3w1`;trainer 需要知道权重是否量化、是否 padding 等,获取这些信息在多数情况下并不简单。

## 提出的方案
- 作者已做传输引擎探索,实现上使用 [nixl](https://github.com/ai-dynamo/nixl) 做内存读写;方案按主用例分 PUSH 与 PULL 两条路径。
- **1. PUSH**(取自 prime-rl #2326 的设计;初步设计重正确性而非性能):orchestrator 协调——trainer rank 0 touch STABLE → orchestrator `/pause` → inference ack all → touch NIXL_READY → trainer 见 NIXL_READY → 全 trainer rank `dist.barrier()` → 每 rank RDMA WRITE → trainer+inference `spg.barrier()` → `/resume`。依赖 trainer 与 inference 预交换 metadata(远端指针地址与长度等)。
- **Proposition 1**:文档化 checkpoint→kernel 的完整转换,如 `ConversionSpec("mlp.gate_up_proj.weight", ("mlp.gate_proj.weight", "mlp.up_proj.weight"), quantization=QuantizationSpec(torch.float8_e4m3fn, ".weight_scale_inv"))` 这类规格(扩展到处理中的所有用法,细节开放讨论)。缺点:需大幅改动 vLLM 整个模型处理逻辑——checkpoint 张量之上的每个操作都要被记录进一个操作图,RL 框架须 replay 该图以构造匹配 vLLM 格式的权重;优点:相当 framework agnostic,RL 框架扩展 vLLM workers 用自选 RDMA 库初始化,就绪后向预交换的张量指针提交写即可。
- **2. PULL**:让 vLLM weight loading RDMA-aware——replay 现有完整加载逻辑,但读远端张量而非本地张量。缺点:待加载张量须预注册在静态地址(变更需额外 metadata 交换)、trainer 侧需以 checkpoint 格式存权重的 scratchpad/参数服务器、vLLM 被锁定到单一 RDMA 框架;优点:trainer 侧更省事。
- 作者强烈倾向 PUSH 语义 + Proposition 1,同时开放讨论。

## 优化点与收益
正文照抄数字(均为作者探索/设计数据,未附完整基准条件):探索实现把 800B 模型的权重传输从 ~40-50sec 缩到 1sec;PUSH 初步设计(含大量同步、以正确性为先)在 ~200+ GPUs 规模上把传输缩到 1.5sec;对照基线为 NCCL broadcast——最多利用 1 张 NIC(50GB/s)。

## 现状与落地
已关闭(关闭于 2026-08-28),labels 含 stale。Feedback Period 一栏为 No response;正文为 WIP 提案,两条路径与 Proposition 1 均标注 open to discussion,作者倾向 PUSH;未列出实现 PR,是否落地需查关联 PR(prime-rl #2326 为其设计出处)。

## 关联
- NIXL(内存读写传输层): https://github.com/ai-dynamo/nixl
- prime-rl 设计 PR: https://github.com/PrimeIntellect-ai/prime-rl/pull/2326

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
