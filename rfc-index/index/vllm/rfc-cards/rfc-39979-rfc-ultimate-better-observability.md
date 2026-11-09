# RFC #39979: [RFC]: Ultimate Better Observability.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39979](https://github.com/vllm-project/vllm/issues/39979) |
| 作者 | noooop |
| 创建日期 | 2026-04-16 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 11 · 💬 11 |
| 分类 | Other |

## 一句话结论
提议构建基于 OpenTelemetry 的多层级粗粒度追踪器,提供 per-request / per-iteration 耗时,适合线上系统监控。

## 提案动机
作者急需比 PyTorch Profiler 粗粒度的追踪器,提供 per-request 与 per-iteration(engine step)耗时用于端到端性能优化。现状:PyTorch Profiler 只能监控少量请求、不适合在线系统;它按单个 PyTorch op 粒度工作,开销显著,而线上系统监控只需要粗得多的粒度。本 RFC 续接 #3616「Even Better Observability」的目标。

## 提出的方案
- 采用 OpenTelemetry 追踪(vLLM 已集成);当前只有 inference_time 追踪,需要更细的耗时分解
- 新追踪器形态类似 PyTorch Profiler,但由多层级自定义粒度 events 构成;追踪层级可像日志级别(DEBUG/INFO/WARNING/…)一样经参数指定;统计指标可从不同追踪层级聚合
- step 1:entrypoints 层——流水线分为(多模态模型的)下载、预处理、engine_call、后处理 3-4 个阶段,作为最粗粒度层级;它不影响核心引擎,可安全在生产环境开启
- level n:runner 层——复用 #24265 引入的自定义 scope(preprocess、forward、postprocess、bookkeep、draft);中间层(level …)正文标注 WIP
- 单个 PyTorch op 粒度仍建议交给 PyTorch Profiler

## 优化点与收益
正文未给出量化数字(定性:粗粒度方案相比 op 级 profiler 开销低,适合线上监控;op 级仍用 PyTorch Profiler)。

## 现状与落地
open(labels 含 keep-open),方案主体为 WIP,仅 entrypoints 层给出具体阶段划分;等待社区进一步讨论补充中间层级。

## 关联
- https://github.com/vllm-project/vllm/issues/3616
- https://github.com/vllm-project/vllm/issues/40781
- https://github.com/vllm-project/vllm/issues/39438
- https://github.com/vllm-project/vllm/issues/32573
- https://github.com/vllm-project/vllm/issues/32162

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
