# RFC #24945: [RFC] SGLang EPD Performance / Architecture / Observability Enhancements

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24945](https://github.com/sgl-project/sglang/issues/24945) |
| 作者 | Abatom |
| 创建日期 | 2026-05-11 |
| 状态 | ⚪ closed (2026-09-07) |
| labels | (none) |
| 热度 | 👍 13 · 💬 1 |
| 分类 | Architecture / Engine |

## 一句话结论
把内部生产验证过的 EPD(Encoder-Prefill-Decode)性能、架构与可观测性优化重写后增量回馈上游。

## 提案动机
EPD 分离由 #12263 于 2025 年 12 月引入社区;作者团队 2026 年 1 月把它移植进基于 v0.5.5 的内部 fork,围绕 MiMo-V2.5 做了大量工程优化与稳定性修复,并经大规模生产环境长时间在线推理验证。这些改动尚未回馈上游,本 RFC 的目标是按当前 mainline 重写并增量交付。

## 提出的方案
正文逐条列出已解决的瓶颈与做法:
- 大图场景下 CPU 端 resize/normalize/patchify 主导端到端延迟 → 预处理迁移到 GPU(社区在 #15456 独立实现了同一方案)。
- 高 QPS 下图片逐张串行下载、PIL 大图解码受 GIL 限制 → 用 multiprocessing 多进程并行下载 + 解码绕开 GIL(社区在 #15456 用多线程做了类似能力)。
- 下载与前向串行(batch N 前向时 batch N+1 只能等)→ 建「多进程下载+解码 → 队列 → GPU 预处理+ViT 前向」流水线,并发执行、跨 batch 重叠。
- 小 batch、固定形状下 ViT/Audio 小算子 kernel launch 开销占比高 → CUDA Graph capture + replay(把 Qwen 系已有的 ViTCudaGraphRunner 扩展到 MiMo ViT,新增 Audio CUDA Graph 路径)。
- Decode 节点重复下载图片/视频/音频并重跑 processor → Prefill 完成后经 bootstrap server 向 Decode 转发轻量 metadata,跳过整条下载+预处理路径。
- 另有 per-process DP Encoder、SHM 传输增强、视频音轨自动抽取、MiMo-V2.5 EPD 适配、OTel tracing、MM token 传播、EncoderMetricsCollector、HTTP 会话复用、细粒度多模态缓存、RadixCache.split_by_offset、消除 CUDA 同步等。
正文附 16 项 PR 清单(feature/performance/architecture/observability 分类)。

## 优化点与收益
正文给出 standalone encoder benchmark(MiMo-V2.5,1K-token 图像):QPS 15 → 30,Avg latency 78.39 ms → 80.28 ms,P90 latency 100.76 ms → 82.94 ms;正文原话「QPS improved 2× with comparable latency」,收益主要来自跨请求 batching、并行图片下载、下载/前向流水线与 GPU 侧预处理。基准工具已作为独立 PR #24700 提交。

## 现状与落地
已关闭(2026-09-07)。PR 清单中多项标注 merged(#24931、#25588、#25964、#26576、#28056、#26082、#27122、#27564 等),另有若干项(#29712、#28435、#26708 等)未标注状态;整体落地情况需查关联 PR。

## 关联
- #12263(EPD 引入): https://github.com/sgl-project/sglang/issues/12263
- #15456(社区同方案): https://github.com/sgl-project/sglang/issues/15456
- #24700(基准工具 PR): https://github.com/sgl-project/sglang/issues/24700
- #24931(MiMo-V2.5 EPD 支持,merged): https://github.com/sgl-project/sglang/issues/24931

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
