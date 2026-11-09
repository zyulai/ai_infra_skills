# RFC #50913: [RFC]: Cutting P/D-disaggregation TTFT: first-token seeding, fast KV side channels, and concurrent router dispatch (NixlConnector)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50913](https://github.com/vllm-project/vllm/issues/50913) |
| 作者 | dczhu |
| 创建日期 | 2026-08-03 |
| 状态 | 🟢 open |
| labels | RFC, quantization |
| 热度 | 👍 1 · 💬 12 |
| 分类 | Disaggregated Serving |

## 一句话结论
针对 NixlConnector 3P1D 部署,用首 token 播种、ZMQ 快速旁路通道与 router 并发派发,把 P→D KV 交接关键路径从 ~197ms 压到 ~36ms。

## 提案动机
范围是 vLLM V1 engine(NixlConnector P/D 路径)+ vllm-project/router。在单节点 3P1D 部署中,P→D 的 KV 交接在「编排」上的开销远大于「数据传输」本身。在 8×L20(Qwen3-32B-FP8, TP2, 3P1D, fp8 KV, FlashInfer, ~4.9k-token prompts)上追踪,基线交接 p50 分解为:D 在 KV 到达后重算 token#1 58.5ms(P 已采样但 router 丢弃)、D worker 晚一步启动 NIXL pull 28.8ms(元数据搭 SchedulerOutput、排在 in-flight forward 之后)、D scheduler 晚一步得知完成 28.1ms、D 前端(HTTP/tokenize/enqueue/schedule)在 P 结束后才串行运行 32.2ms、NIXL 传输窗口 46.9ms(其中纯传输约 33ms)。此外默认 `--block-size 16` 下 ~5k-token 的 pull 会碎裂成数万个 ~8KB NIXL 描述符,每描述符开销主导:传输窗口 ~105ms、交接 ~263ms。

## 提出的方案
- Proposal 0(无代码改动):将 `--block-size` 从 16 调到 64(FlashInfer 的有效页上限)并写入部署指南;传输窗口缩短 ~58ms,端到端交接降幅更大(~66ms,到 197ms)。
- P1 — First-token seeding:P 的 `NixlConnector.request_finished()` 把采样出的 token 放进 `kv_transfer_params.first_token_ids`;D 的 KV pull 完成后 scheduler 直接播种(`append_output_token_ids`)并立即发出,同时钳制 `num_computed_tokens` 使下一步前向直接产出 token#2。消除 ~58ms 重算。Flag:`VLLM_PD_SEED_FIRST_TOKEN`。
- P2 — Fast KV side channels:D 的 EngineCore 与 TP worker 之间加两条 ZMQ IPC 通道:fast dispatch 在 `schedule()` 后立刻发布 `reqs_to_recv` 让 worker 监听线程立即发起 NIXL 读;fast notify 由 worker poller 在 ~0.5ms 内检测完成并推给 EngineCore bridge 线程聚合各 TP rank、播种 token#1、直写输出队列。消除两个 ~28ms 的 step 边界等待。Flags:`VLLM_PD_NIXL_FAST_DISPATCH`、`VLLM_PD_NIXL_FAST_NOTIFY`。
- P3 — Concurrent router dispatch + D-side prepare parking:router 并发派发 P 和 D;D 先收到 prepare 请求,完成 tokenize、入队、前缀匹配并分配 block(「武装」)后进入 `WAITING_FOR_REMOTE_KVS`;P 完成后 router 把 `kv_transfer_params` POST 到 D 的新端点 `POST /v1/pd_kv_ready`,直推 P2 bridge 线程合并参数并快速派发 pull,ready→pull 路径不经过任何 step 边界。丢失通知走 `VLLM_PD_PREPARE_READY_TIMEOUT_S` 超时并回退到现有失败加载路径(本地重算)。Flag(router):`VLLM_PD_CONCURRENT_DISPATCH`。
- P3a — Early-arm 作为内存/性能权衡:park 时不分配(late-alloc)几乎无收益(step 边界等待只是转移到 ready 路径,~15ms p50);park 时即分配(early-arm)能兑现收益,但会在 prefill 期间(其负载下约 0.7s)钉住该请求的 KV block。提供 flag `VLLM_PD_EARLY_ARM`,late-alloc 作为内存受限时的回退。
- 所有改动均由 flag 门控,可回退到现有行为。

## 优化点与收益
测试条件:单节点 3P1D、NixlConnector、8×L20、Qwen3-32B-FP8、TP2、fp8 KV、FlashInfer、~4.9k-token prompts、1000 请求。交接关键路径(P 发出 token#1 → 客户端收到 token#1)p50:Stock NixlConnector `--block-size 16`(默认)262.6ms → `--block-size 64` 197.1ms → +P1+P2 63.2ms → +P3(late-alloc)51.8ms → +P3a(early-arm)35.7ms,即 ~197ms → ~36ms p50。相对默认 stock NIXL 3P1D 配置:mean TTFT 863.9ms(−23%)、median 783.1ms(−23%)、P99 TTFT 2296.1ms(−10%)、吞吐 1.23 req/s(+7%)、TPOT 30.2ms(不变,无回归)。

## 现状与落地
仍在讨论中(open)。正文给出了完整的基线拆解、逐项 proposal 与实测结果(基线 vLLM `b17039bc` v0.14.0,router `ce2d5133` v0.1.12),各项改动 flag 门控;合入进度需查后续 PR。注:pack 正文在 20k 字符处截断,本卡基于现有内容(Scope/Summary/Problem/Proposed changes/Results/References 各节)生成。

## 关联
- vllm-project/router(正文引用其 P/D 路由)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
