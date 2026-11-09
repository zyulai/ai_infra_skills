# RFC #23206: [RFC] Sglang non-GPU process rust migration

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23206](https://github.com/sgl-project/sglang/issues/23206) |
| 作者 | rainj-me |
| 创建日期 | 2026-04-20 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 12 · 💬 15 |
| 分类 | Other |

## 一句话结论
把 SGLang 请求前半段(HTTP 服务、tokenizer/detokenizer 等)从 Python 迁移到单进程多线程 Rust,保持 API 与行为不变。

## 提案动机
当前请求前半部分(网络入口到 tokenized request 交给 GPU scheduler 为止)由 Python 的 tokenizer/detokenizer manager 与 HTTP server 承担。计划将其替换为单进程、thread-per-core 的 Rust 实现,拥有到 scheduler 提交为止的请求生命周期;Rust server 须是 Python 版的 drop-in(相同 endpoints、请求/响应 schema、streaming 语义),可挂 flag 后与 Python 做 A/B。范围包含 HTTP(S)/gRPC/WS/h2/h3 server、request FSM、tokenizer/detokenizer manager、多模态 encoder dispatch、控制 API、大张量零拷贝 handoff、跨 DP 组 dispatch;不含 model kernels、attention backends、KV-cache internals、量化,保留现有 Python scheduler 与 GPU worker。

## 提出的方案
- 双 FSM 一进程:ingress FSM(Received→校验/编码/分词→Queued)与 egress FSM(Streaming 自环→Finalizing→Completed),由同一 core 持有,在 ring-buffer/scheduler 边界会合,无共享锁。
- 六个线程组:api-server(async runtime)、tokenizer-manager(跑 FSM)、encoder、tokenizer、detokenizer(core-bound)与 Python scheduler(main thread);组间只经 channel 通信。
- 通道与零拷贝:bounded MPSC(TokenizerManager→Scheduler,try_send 做背压)+ per-core SPSC 返回;Rust→Python 边界用 msgpack 序列化,只有不可变字节跨边界,GIL 不进热路径;张量数据走 pinned memory 缓冲池 slot handle,不序列化、不拷贝。
- DP attention:每 DP 组首个 rank 运行无 api-server 的 Rust server;单个 api-server 把控制消息复制广播到各 DP 组,generation 请求 round-robin 分发。
- 推进方式:每阶段挂 flag 灰度(shadow → 小比例 → 放量),与 Python 的差分测试(同输入同 token/frame)作为首要正确性闸门;注意 io_uring 对内核版本的最低要求并保留 epoll+eventfd 回退。
- Phase 1 目标 2026-07-17:29 项工作分解中,pickle→msgpack(PR #28688 已合)、单进程框架、tok/detok manager、/generate+streaming、PD disaggregation、DP attention 等已完成,encoder 与 metrics WIP,API 全量对齐等 pending。

## 优化点与收益
正文未给出量化数字。收益定位为工程性:替换 Python 前端实现,并把 Rust↔Python 边界收窄为 handle 而非数据以避免 GIL 争用;性能数字待 benchmark_serving 优化项(状态 WIP)产出。

## 现状与落地
仍在推进(open)。计划文档标注 Status: Draft(2026-06-18 更新),Phase 1 core 里程碑定在 2026-07-17;正文同时提醒:该日期约 4 周时基础项 #4 尚未启动,若不把 PD disaggregation、DP attention、统一协议库移出里程碑,全量交付并不现实。

## 关联
- PR #29799(issue 关联 PR)
- Roadmap #22949
- PoC PR #23360(Closed)
- PR #28688(pickle→msgpack,Merged)
- Followup #29465

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
